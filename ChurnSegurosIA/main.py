from fastapi import FastAPI
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression

from ChurnSegurosIA.models.prediccion_request import PrediccionRequest

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
# Modelo IA (persistente)
# =========================
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = LogisticRegression()
    model.fit([[0, 0, 0, 0, 0]], [0])  # inicio mínimo
    joblib.dump(model, MODEL_PATH)

# =========================
# Endpoints
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(request: PrediccionRequest):

    # 1. Obtener datos necesarios desde la base de datos
    with engine.begin() as conn:

        cliente = conn.execute(text("""
            SELECT fecha_nacimiento, fecha_alta, nombre
            FROM dbo.clientes
            WHERE cliente_id = :cliente_id
        """), {"cliente_id": request.cliente_id}).fetchone()

        producto = conn.execute(text("""
            SELECT nombre, prima_base
            FROM dbo.productos
            WHERE producto_id = :producto_id
        """), {"producto_id": request.producto_id}).fetchone()

        reclamos = conn.execute(text("""
            SELECT COUNT(*)
            FROM dbo.reclamos
            WHERE cliente_id = :cliente_id
              AND producto_id = :producto_id
        """), {
            "cliente_id": request.cliente_id,
            "producto_id": request.producto_id
        }).scalar()

    # 2. Calcular variables derivadas
    hoy = pd.Timestamp.now().date()

    edad = (hoy - cliente.fecha_nacimiento).days // 365
    antiguedad = (hoy - cliente.fecha_alta).days // 365

    # 3. Construir vector de entrada para la IA
    df = pd.DataFrame([{
        "antiguedad": antiguedad,
        "num_siniestros": reclamos,
        "prima_mensual": producto.prima_base,
        "uso_contacto": reclamos,
        "reclamos": reclamos
    }])

    # 4. Predicción
    prob = model.predict_proba(df)[0][1]
    prob_percent = round(prob * 100, 2)

    # 5. Guardar predicción en la base
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO dbo.predictions
            (cliente_id, producto_id, probabilidad_cancelacion)
            VALUES (:cliente_id, :producto_id, :prob)
        """), {
            "cliente_id": request.cliente_id,
            "producto_id": request.producto_id,
            "prob": float(prob)
        })

    # 6. Clasificar nivel de riesgo
    if prob_percent >= 80:
        nivel_riesgo = "ALTO"
    elif prob_percent >= 40:
        nivel_riesgo = "MEDIO"
    else:
        nivel_riesgo = "BAJO"

    # 7. Respuesta final orientada a negocio
    return {
        "cliente": cliente.nombre,
        "producto": producto.nombre,
        "probabilidad_cancelacion": prob_percent,
        "nivel_riesgo": nivel_riesgo
    }