import pandas as pd
import numpy as np

data = pd.DataFrame({
    "antiguedad": [1, 2, 5, 8, 10, 3, 4, 7, 9, 6],
    "num_siniestros": [3, 2, 1, 0, 0, 2, 2, 1, 0, 1],
    "prima_mensual": [120, 110, 90, 80, 75, 100, 105, 85, 70, 95],
    "uso_contacto": [8, 6, 3, 1, 0, 5, 6, 2, 1, 4],
    "reclamos": [2, 1, 0, 0, 0, 1, 1, 0, 0, 1],
    "churn": [1, 1, 0, 0, 0, 1, 1, 0, 0, 0]
})

print(data)

X = data.drop("churn", axis=1)
y = data["churn"]

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)


cliente_nuevo = pd.DataFrame(
    [[4, 1, 95, 5, 1]],
    columns=X.columns)

probabilidades = model.predict_proba(cliente_nuevo)
print("Probabilidad de cancelar:", probabilidades[0][1])