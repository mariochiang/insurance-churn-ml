from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os
import logging

from sqlalchemy import create_engine, text, bindparam
from sqlalchemy.exc import SQLAlchemyError
from sklearn.linear_model import LogisticRegression

from models.prediccion_request import PrediccionRequest


# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================
# Configuracion Base de Datos Azure SQL
# =========================
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL no esta configurada")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=1800,
)


# =========================
# App FastAPI
# =========================
app = FastAPI(title="Churn Seguros IA API")


# =========================
# Modelo IA
# =========================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURE_COLUMNS = [
    "antiguedad",
    "num_siniestros",
    "prima_mensual",
    "uso_contacto",
    "reclamos",
]

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logger.info("Modelo cargado desde %s", MODEL_PATH)
else:
    logger.warning(
        "No se encontro model.pkl. Se usara un modelo dummy solo para desarrollo."
    )

    model = LogisticRegression()

    X_dummy = pd.DataFrame(
        [
            {
                "antiguedad": 0,
                "num_siniestros": 0,
                "prima_mensual": 0,
                "uso_contacto": 0,
                "reclamos": 0,
            },
            {
                "antiguedad": 10,
                "num_siniestros": 5,
                "prima_mensual": 100000,
                "uso_contacto": 5,
                "reclamos": 5,
            },
        ],
        columns=FEATURE_COLUMNS,
    )

    y_dummy = [0, 1]
    model.fit(X_dummy, y_dummy)


# =========================
# Helpers
# =========================
def calcular_antiguedad(fecha_alta):
    if not fecha_alta:
        return 0

    hoy = pd.Timestamp.now().date()
    fecha_alta_date = pd.Timestamp(fecha_alta).date()

    antiguedad = (hoy - fecha_alta_date).days // 365

    return max(antiguedad, 0)


def crear_dataframe_prediccion(antiguedad, reclamos, prima_base):
    return pd.DataFrame(
        [
            {
                "antiguedad": antiguedad,
                "num_siniestros": reclamos,
                "prima_mensual": float(prima_base or 0),
                "uso_contacto": reclamos,
                "reclamos": reclamos,
            }
        ],
        columns=FEATURE_COLUMNS,
    )


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PrediccionRequest):
    rut = request.rut.strip()

    if not rut:
        raise HTTPException(status_code=400, detail="El RUT es obligatorio")

    try:
        with engine.begin() as conn:
            cliente = conn.execute(
                text(
                    """
                    SELECT cliente_id, nombre, fecha_nacimiento, fecha_alta
                    FROM dbo.clientes
                    WHERE rut = :rut
                    """
                ),
                {"rut": rut},
            ).fetchone()

            if not cliente:
                raise HTTPException(status_code=404, detail="Cliente no encontrado")

            productos = conn.execute(
                text(
                    """
                    SELECT p.producto_id, p.nombre, p.prima_base
                    FROM dbo.clientes_productos cp
                    JOIN dbo.productos p ON cp.producto_id = p.producto_id
                    WHERE cp.cliente_id = :cliente_id
                      AND cp.activo = 1
                    """
                ),
                {"cliente_id": cliente.cliente_id},
            ).fetchall()

            if not productos:
                return {
                    "cliente": cliente.nombre,
                    "rut": rut,
                    "productos_evaluados": 0,
                    "tiene_riesgo_alto": False,
                    "productos_en_riesgo": [],
                }

            producto_ids = [producto.producto_id for producto in productos]

            query_reclamos = text(
                """
                SELECT producto_id, COUNT(*) AS total_reclamos
                FROM dbo.reclamos
                WHERE cliente_id = :cliente_id
                  AND producto_id IN :producto_ids
                GROUP BY producto_id
                """
            ).bindparams(bindparam("producto_ids", expanding=True))

            reclamos_rows = conn.execute(
                query_reclamos,
                {
                    "cliente_id": cliente.cliente_id,
                    "producto_ids": producto_ids,
                },
            ).fetchall()

            reclamos_por_producto = {
                row.producto_id: row.total_reclamos for row in reclamos_rows
            }

            antiguedad = calcular_antiguedad(cliente.fecha_alta)
            resultados_alto_riesgo = []
            predicciones_guardadas = []

            for producto in productos:
                reclamos = reclamos_por_producto.get(producto.producto_id, 0)

                df = crear_dataframe_prediccion(
                    antiguedad=antiguedad,
                    reclamos=reclamos,
                    prima_base=producto.prima_base,
                )

                prob = model.predict_proba(df)[0][1]
                prob_percent = round(float(prob) * 100, 2)

                conn.execute(
                    text(
                        """
                        INSERT INTO dbo.predictions
                        (cliente_id, producto_id, probabilidad_cancelacion)
                        VALUES (:cliente_id, :producto_id, :prob)
                        """
                    ),
                    {
                        "cliente_id": cliente.cliente_id,
                        "producto_id": producto.producto_id,
                        "prob": float(prob),
                    },
                )

                predicciones_guardadas.append(
                    {
                        "producto": producto.nombre,
                        "probabilidad_cancelacion": prob_percent,
                    }
                )

                if prob_percent >= 80:
                    resultados_alto_riesgo.append(
                        {
                            "producto": producto.nombre,
                            "probabilidad_cancelacion": prob_percent,
                            "nivel_riesgo": "ALTO",
                        }
                    )

        return {
            "cliente": cliente.nombre,
            "rut": rut,
            "productos_evaluados": len(productos),
            "tiene_riesgo_alto": len(resultados_alto_riesgo) > 0,
            "productos_en_riesgo": resultados_alto_riesgo,
            "predicciones": predicciones_guardadas,
        }

    except HTTPException:
        raise

    except SQLAlchemyError:
        logger.exception("Error consultando la base de datos")
        raise HTTPException(
            status_code=500,
            detail="Error consultando la base de datos",
        )

    except Exception:
        logger.exception("Error inesperado generando prediccion")
        raise HTTPException(
            status_code=500,
            detail="Error inesperado generando prediccion",
        )
