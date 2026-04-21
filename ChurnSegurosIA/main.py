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

    resultados_alto_riesgo = []

    with engine.begin() as conn:

        # 1. Obtener cliente por RUT
        cliente = conn.execute(text("""
            SELECT cliente_id, nombre, fecha_nacimiento, fecha_alta
            FROM dbo.clientes
            WHERE rut = :rut
        """), {"rut": request.rut}).fetchone()

        # 2. Obtener todos los productos activos del cliente
        productos = conn.execute(text("""
            SELECT p.producto_id, p.nombre, p.prima_base
            FROM dbo.clientes_productos cp
            JOIN dbo.productos p ON cp.producto_id = p.producto_id
            WHERE cp.cliente_id = :cliente_id
              AND cp.activo = 1
        """), {"cliente_id": cliente.cliente_id}).fetchall()

    hoy = pd.Timestamp.now().date()
    antiguedad = (hoy - cliente.fecha_alta).days // 365

    # 3. Evaluar cada producto
    for producto in productos:

        with engine.begin() as conn:
            reclamos = conn.execute(text("""
                SELECT COUNT(*)
                FROM dbo.reclamos
                WHERE cliente_id = :cliente_id
                  AND producto_id = :producto_id
            """), {
                "cliente_id": cliente.cliente_id,
                "producto_id": producto.producto_id
            }).scalar()

        # Vector IA
        df = pd.DataFrame([{
            "antiguedad": antiguedad,
            "num_siniestros": reclamos,
            "prima_mensual": producto.prima_base,
            "uso_contacto": reclamos,
            "reclamos": reclamos
        }])

        prob = model.predict_proba(df)[0][1]
        prob_percent = round(prob * 100, 2)

        # Guardar predicción
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO dbo.predictions
                (cliente_id, producto_id, probabilidad_cancelacion)
                VALUES (:cliente_id, :producto_id, :prob)
            """), {
                "cliente_id": cliente.cliente_id,
                "producto_id": producto.producto_id,
                "prob": float(prob)
            })

        if prob_percent >= 80:
            resultados_alto_riesgo.append({
                "producto": producto.nombre,
                "probabilidad_cancelacion": prob_percent,
                "nivel_riesgo": "ALTO"
            })

    return {
        "cliente": cliente.nombre,
        "rut": request.rut,
        "tiene_riesgo_alto": len(resultados_alto_riesgo) > 0,
        "productos_en_riesgo": resultados_alto_riesgo
    }