# Manutenção Preditiva com FastAPI, MLflow e Vertex AI

Este projeto implementa uma pipeline de machine learning voltada à **manutenção preditiva**, com:

- **FastAPI** para servir o modelo como API REST.
- **MLflow** para gerenciamento de experimentos e modelos.
- **Google Cloud Vertex AI** para deployment e escalabilidade.
- **Docker** para containerização.

## Estrutura do Projeto

```
cloudProject/
├── download_model.py         # Faz download do modelo treinado via MLflow
├── train_mlflow.ipynb        # Notebook  para treino
├── requirements.txt          # Dependências principais
├── README.md                 # Este arquivo
├── xgboost.png               # Imagem ilustrativa do modelo
├── mlruns/                   # Diretório padrão do MLflow
└── vertex_model/
    ├── app.py               # API criada com FastAPI
    ├── runfile.ipynb        # Testa a API localmente
    ├── Dockerfile           # Build da imagem do container
    ├── requirements.txt     # Requisitos da API (deixei igual o principal)
    ├── model/               # Modelos salvos (.pkl)
    └── mlruns/              # Logs/experimentos locais (se aplicável)
```

## Requisitos

- Python 3.10
- Docker
- Conta no Google Cloud Platform (GCP)
- Vertex AI e Artifact Registry habilitados
- Google Cloud CLI (gcloud)

## Treinamento do Modelo

### Via Notebook
Abra e execute o notebook `train_mlflow.ipynb`.

## Download do Modelo Treinado

Use o script abaixo para recuperar o melhor modelo salvo via MLflow:

```
python download_model.py
```

## Deploy com Vertex AI

### 1. Autenticação no GCP
```
gcloud auth login
gcloud config set project [SEU_PROJECT_ID]
```

### 2. Build da Imagem Docker e Push
```
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/[SEU_PROJECT_ID]/manutencao-api/vertex-model
```

### 3. Upload do Modelo
```
gcloud ai models upload \
  --region=us-central1 \
  --display-name=vertex-model-fastapi \
  --container-image-uri=us-central1-docker.pkg.dev/[SEU_PROJECT_ID]/manutencao-api/vertex-model \
  --container-ports=8080 \
  --container-health-route=/health \
  --container-predict-route=/predict
```

### 4. Criação e Deploy do Endpoint
```
gcloud ai endpoints create \
  --region=us-central1 \
  --display-name=vertex-endpoint

gcloud ai endpoints deploy-model ENDPOINT_ID \
  --model=MODEL_ID \
  --region=us-central1 \
  --display-name=vertex-model-deployed \
  --traffic-split=0=100
```

## Testando a API com curl

```
curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  https://us-central1-aiplatform.googleapis.com/v1/projects/[SEU_PROJECT_ID]/locations/us-central1/endpoints/ENDPOINT_ID:predict \
  -d '{
        "instances": [{
          "Air temperature [K]": 298.1,
          "Process temperature [K]": 308.6,
          "Rotational speed [rpm]": 1550,
          "Torque [Nm]": 43.4,
          "Tool wear [min]": 0,
          "Type": "M"
        }]
      }'
```

## Licença

MIT License
