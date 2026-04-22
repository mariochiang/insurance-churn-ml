from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
from sqlalchemy import create_engine, text
from sklearn.linear_model import LogisticRegression

from models.prediccion_request import PrediccionRequest

# =========================
# Configuración Base de Datos (Azure SQL)
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no está configurada")

engine = create_engine(DATABASE_URL)

# =========================
# App FastAPI
# =========================
app = FastAPI(title="Churn Seguros IA API")

# =========================
# Modelo IA (persistente)
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = LogisticRegression()
    X_dummy = [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]
    y_dummy = [0, 1]
    model.fit(X_dummy, y_dummy)

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
        cliente = conn.execute(
            text("""
                SELECT cliente_id, nombre, fecha_nacimiento, fecha_alta
                FROM dbo.clientes
                WHERE rut = :rut
            """),
            {"rut": request.rut}
        ).fetchone()

        if not cliente:
            raise HTTPException(status_code=404, detail="Cliente no encontrado")

        productos = conn.execute(
            text("""
                SELECT p.producto_id, p.nombre, p.prima_base
                FROM dbo.clientes_productos cp
                JOIN dbo.productos p ON cp.producto_id = p.producto_id
                WHERE cp.cliente_id = :cliente_id
                  AND cp.activo = 1
            """),
            {"cliente_id": cliente.cliente_id}
        ).fetchall()

    hoy = pd.Timestamp.now().date()
    antiguedad = (hoy - cliente.fecha_alta).days // 365

    for producto in productos:

        with engine.begin() as conn:
            reclamos = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM dbo.reclamos
                    WHERE cliente_id = :cliente_id
                      AND producto_id = :producto_id
                """),
                {
                    "cliente_id": cliente.cliente_id,
                    "producto_id": producto.producto_id
                }
            ).scalar()

        df = pd.DataFrame([{
            "antiguedad": antiguedad,
            "num_siniestros": reclamos,
            "prima_mensual": producto.prima_base,
            "uso_contacto": reclamos,
            "reclamos": reclamos
        }])

        prob = model.predict_proba(df)[0][1]
        prob_percent = round(prob * 100, 2)

        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO dbo.predictions
                    (cliente_id, producto_id, probabilidad_cancelacion)
                    VALUES (:cliente_id, :producto_id, :prob)
                """),
                {
                    "cliente_id": cliente.cliente_id,
                    "producto_id": producto.producto_id,
                    "prob": float(prob)
                }
            )

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