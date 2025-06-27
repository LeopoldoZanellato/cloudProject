# download_model.py

import os
import shutil
import joblib

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# 1) Configurações
model_name = "manutencao_modelo_final"
dst_dir = "vertex_model/model"
os.makedirs(dst_dir, exist_ok=True)

# 2) Cliente MLflow
client = MlflowClient()

# 3) Pegar todas as versões registradas
all_versions = client.search_model_versions(f"name = '{model_name}'")

# 4) Filtrar as que estão em Production
prod_versions = [v for v in all_versions if v.current_stage.lower() == "production"]

if prod_versions:
    selected = prod_versions[0]
else:
    # Se não houver nenhuma em Production, pega a de maior número
    selected = sorted(all_versions, key=lambda v: int(v.version))[-1]

model_version = selected.version
run_id = selected.run_id

print(f"⏳ Baixando versão {model_version} (stage='{selected.current_stage}') do modelo '{model_name}'")

# 5) Carrega o modelo do Registry (pela versão numérica)
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
joblib.dump(model, os.path.join(dst_dir, "model.pkl"))

# 6) Baixa os artefatos de pré-processamento daquele run
prep_local = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="preprocessing"
)

# 7) Copia scaler e encoder para a pasta de deploy
shutil.copy(os.path.join(prep_local, "scaler.pkl"), os.path.join(dst_dir, "scaler.pkl"))
shutil.copy(os.path.join(prep_local, "encoder.pkl"), os.path.join(dst_dir, "encoder.pkl"))

print("✅ Arquivos prontos em vertex_model/model/:", os.listdir(dst_dir))
