from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os
from sqlalchemy import create_engine, text

# =========================
# Conexión Azure SQL Server
# =========================
DATABASE_URL = (
    "mssql+pyodbc://sqladmin:Ma1986%2B%2B@sql-insurance-ia01.database.windows.net:1433/"
    "insurance_db"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&Encrypt=yes"
    "&TrustServerCertificate=no"
    "&ConnectionTimeout=30"
)

engine = create_engine(DATABASE_URL)

# =========================
# App
# =========================
app = FastAPI()

# =========================
# Modelo (persistente)
# =========================
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # inicio frío
    model = LogisticRegression()
    model.fit(
        [[0, 0, 0, 0, 0]],
        [0]
    )
    joblib.dump(model, MODEL_PATH)

# =========================
# Schema de entrada
# =========================
class Cliente(BaseModel):
    antiguedad: int
    num_siniestros: int
    prima_mensual: float
    uso_contacto: int
    reclamos: int

# =========================
# Endpoints base
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# Predicción + persistencia
# =========================
@app.post("/predict")
def predict(client: Cliente):
    df = pd.DataFrame([client.dict()])
    prob = model.predict_proba(df)[0][1]

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO dbo.predictions
                (antiguedad, num_siniestros, prima_mensual, uso_contacto, reclamos, probabilidad_cancelacion)
                VALUES
                (:antiguedad, :num_siniestros, :prima_mensual, :uso_contacto, :reclamos, :prob)
            """),
            {
                "antiguedad": client.antiguedad,
                "num_siniestros": client.num_siniestros,
                "prima_mensual": client.prima_mensual,
                "uso_contacto": client.uso_contacto,
                "reclamos": client.reclamos,
                "prob": float(prob)
            }
        )

    return {"probabilidad_cancelacion": round(prob * 100, 2)}

# =========================
# Consulta histórica
# =========================
@app.get("/predictions")
def get_predictions(limit: int = 20):
    with engine.begin() as conn:
        result = conn.execute(
            text("""
                SELECT TOP (:limit)
                    id,
                    antiguedad,
                    num_siniestros,
                    prima_mensual,
                    uso_contacto,
                    reclamos,
                    ROUND(probabilidad_cancelacion * 100, 2) AS probabilidad_percent,
                    created_at
                FROM dbo.predictions
                ORDER BY created_at DESC
            """),
            {"limit": limit}
        )
        return [dict(row._mapping) for row in result]

# =========================
# Segmentación de riesgo
# =========================
@app.get("/risk-segments")
def risk_segments():
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT
                CASE
                    WHEN probabilidad_cancelacion >= 0.8 THEN 'ALTO'
                    WHEN probabilidad_cancelacion >= 0.4 THEN 'MEDIO'
                    ELSE 'BAJO'
                END AS nivel_riesgo,
                COUNT(*) AS cantidad
            FROM dbo.predictions
            GROUP BY
                CASE
                    WHEN probabilidad_cancelacion >= 0.8 THEN 'ALTO'
                    WHEN probabilidad_cancelacion >= 0.4 THEN 'MEDIO'
                    ELSE 'BAJO'
                END
        """))
        return [dict(row._mapping) for row in result]

# =========================
# Perfil típico alto riesgo
# =========================
@app.get("/high-risk-profile")
def high_risk_profile():
    with engine.begin() as conn:
        result = conn.execute(text("""
            SELECT
                AVG(antiguedad) AS avg_antiguedad,
                AVG(num_siniestros) AS avg_siniestros,
                AVG(prima_mensual) AS avg_prima,
                AVG(uso_contacto) AS avg_contacto,
                AVG(reclamos) AS avg_reclamos
            FROM dbo.predictions
            WHERE probabilidad_cancelacion >= 0.8
        """))
        return dict(result.fetchone()._mapping)

# =========================
# Reentrenamiento real
# =========================
@app.post("/retrain-from-db")
def retrain_from_db():
    df = pd.read_sql("""
        SELECT
            antiguedad,
            num_siniestros,
            prima_mensual,
            uso_contacto,
            reclamos,
            probabilidad_cancelacion
        FROM dbo.predictions
    """, engine)

    if len(df) < 5:
        return {"error": "No hay suficientes datos"}

    X_new = df.drop("probabilidad_cancelacion", axis=1)
    y_new = (df["probabilidad_cancelacion"] >= 0.5).astype(int)

    model.fit(X_new, y_new)
    joblib.dump(model, MODEL_PATH)

    return {
        "status": "model retrained",
        "rows_used": len(df)
    }