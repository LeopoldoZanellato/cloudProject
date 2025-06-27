from fastapi import FastAPI, Request
import joblib, pandas as pd, numpy as np

app = FastAPI()

# Health checks
@app.get("/", include_in_schema=False)
async def root():
    return {"status": "ok"}

@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}

# Carregar modelo e pré-processadores
model = joblib.load("model/model.pkl")
scaler = joblib.load("model/scaler.pkl")
encoder = joblib.load("model/encoder.pkl")

# Definir colunas esperadas
numerical_columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_columns = ['Type']

@app.post("/predict")
async def predict(request: Request):
    body = await request.json()

    # 1) tenta ler o formato local { "data": […] }
    payload = body.get("data")
    if payload is None:
        # 2) senão, extrai de [{ "data": […] }] ou só de instances
        inst = body.get("instances", [])
        if isinstance(inst, list) and inst and isinstance(inst[0], dict) and "data" in inst[0]:
            payload = inst[0]["data"]
        else:
            payload = inst

    # Constrói o DataFrame a partir da lista de dicionários
    data = pd.DataFrame(payload)

    # Pré-processamento
    X_num = scaler.transform(data[numerical_columns])
    X_cat = encoder.transform(data[categorical_columns])
    X_processed = np.hstack([X_num, X_cat])

    # Predição
    preds = model.predict(X_processed)
    return {"predictions": preds.tolist()}
