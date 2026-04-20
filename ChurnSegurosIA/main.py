from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# =========================
# Base de datos (preparada)
# =========================
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql+psycopg2://usuario:password@localhost:5432/segurodb"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# =========================
# App
# =========================
app = FastAPI()

# =========================
# Datos de entrenamiento (temporales)
# =========================
data = pd.DataFrame({
    "antiguedad": [1, 2, 5, 8, 10, 3, 4, 7, 9, 6],
    "num_siniestros": [3, 2, 1, 0, 0, 2, 2, 1, 0, 1],
    "prima_mensual": [120, 110, 90, 80, 75, 100, 105, 85, 70, 95],
    "uso_contacto": [8, 6, 3, 1, 0, 5, 6, 2, 1, 4],
    "reclamos": [2, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    "churn": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
})

X = data.drop("churn", axis=1)
y = data["churn"]

# =========================
# Persistencia del modelo
# =========================
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

# =========================
# Schemas
# =========================
class Cliente(BaseModel):
    antiguedad: int
    num_siniestros: int
    prima_mensual: float
    uso_contacto: int
    reclamos: int

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(client: Cliente):
    df = pd.DataFrame([client.dict()], columns=X.columns)
    prob = model.predict_proba(df)[0][1]
    return {"probabilidad_cancelacion": float(prob)}

@app.post("/train")
def train():
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return {"status": "model retrained"}
