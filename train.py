#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier  # Bônus se quiser testar também
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


bucket_path = 'gs://predictive-maintenance-leopoldo/manutpred.csv'
df_raw = pd.read_csv(bucket_path, storage_options={'token': 'cloud'})


# In[4]:


df = df_raw.copy()
df.head()


# In[5]:


drop_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
df.drop(drop_columns, axis=1, inplace=True)


# In[6]:


df.head()


# In[7]:


# divisão das colunas
categorical_columns = ['Type']
numerical_columns = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
target_column = ['Machine failure']

# colunas que não importam para o modelo
Xdrop_columns = ['UDI', 'Product ID'] 

# Separando features e target
X = df.drop(target_column, axis=1)
X.drop(Xdrop_columns, axis = 1, inplace =True)
y = df[target_column]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[ ]:





# In[8]:


X_train.head()


# In[9]:


y_test.head()


# In[10]:


# CATEGORICAL --> criando o encoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# encoder + df de treino
encoded_type_train = encoder.fit_transform(X_train[categorical_columns])
encoded_train_df = pd.DataFrame(
    encoded_type_train,
    columns=encoder.get_feature_names_out(input_features=categorical_columns),
    index=X_train.index
)


# In[11]:


# CATEGORICAL --> encoder + df de teste
encoded_type_test = encoder.transform(X_test[categorical_columns])

encoded_test_df = pd.DataFrame(
    encoded_type_test,
    columns=encoder.get_feature_names_out(input_features=categorical_columns),
    index=X_test.index
)


# In[12]:


# NUMERICAL --> Scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
#scaler = RobustScaler()
X_train_num = scaler.fit_transform(X_train[numerical_columns])
X_test_num = scaler.transform(X_test[numerical_columns])


# In[13]:


X_train_processed = pd.DataFrame(
    np.hstack([X_train_num, encoded_train_df]),
    columns=numerical_columns + list(encoder.get_feature_names_out(categorical_columns)),
    index=X_train.index
)


# In[14]:


X_test_processed = pd.DataFrame(
    np.hstack([X_test_num, encoded_test_df]),
    columns=numerical_columns + list(encoder.get_feature_names_out(categorical_columns)),
    index=X_test.index
)


# In[15]:


X_train_processed.head()


# In[16]:


def clean_column_names(df):
    df.columns = [
        col.replace(' ', '_')
           .replace('[','')
           .replace(']','')
           .replace('(','')
           .replace(')','')
           .replace('/','_per_')  # Caso tenha barras ou outros símbolos no futuro
        for col in df.columns
    ]
    return df

# Aplicando no treino e teste
X_train_processed = clean_column_names(X_train_processed.copy())
X_test_processed = clean_column_names(X_test_processed.copy())


# In[17]:


X_test_processed.head()


# In[18]:


# Aplica SMOTE apenas no conjunto de treino
smote = SMOTE(random_state=42)
X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)

# (opcional) checar se balanceou
print("Distribuição após SMOTE:")
print(y_train.value_counts())


# In[21]:


models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
}


# Scorers personalizados
scoring = {
    'accuracy': 'accuracy',
    'f1': make_scorer(f1_score, pos_label=1),
    'precision': make_scorer(precision_score, pos_label=1),
    'recall': make_scorer(recall_score, pos_label=1)
}

#mlflow.set_experiment("manutencao_preditiva_multimodel_cv_v2")

mlflow.set_experiment("manutencao_preditiva_multimodel_cv_v2")

for model_name, model in models.items():
    with mlflow.start_run(run_name=f"{model_name}_CV"):

        # Cross-Validation com 5 folds
        results = cross_validate(
            model,
            X_train_processed,
            y_train.values.ravel(),
            cv=5,
            scoring=scoring,
            return_train_score=False
        )

        # Log de métricas médias
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("cv_accuracy", results['test_accuracy'].mean())
        mlflow.log_metric("cv_f1_score", results['test_f1'].mean())
        mlflow.log_metric("cv_precision_class_1", results['test_precision'].mean())
        mlflow.log_metric("cv_recall_class_1", results['test_recall'].mean())

        # Treina no full train e salva o modelo final
        model.fit(X_train_processed, y_train.values.ravel())
        mlflow.sklearn.log_model(model, model_name)



        # Salvar o encoder
        encoder_path = f"encoder.pkl"
        joblib.dump(encoder, encoder_path)
        mlflow.log_artifact(encoder_path, artifact_path="preprocessing")
        os.remove(encoder_path)  # <-- remove o arquivo local


        # Salvar o scaler
        scaler_path = f"scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        os.remove(scaler_path)  # <-- remove o arquivo local
        
        # Tags explicativas
        mlflow.set_tag("preprocessing_scaler", scaler.__class__.__name__)
        mlflow.set_tag("preprocessing_encoder", encoder.__class__.__name__)
        
        print(f"{model_name} → CV F1: {results['test_f1'].mean():.4f} | CV Acc: {results['test_accuracy'].mean():.4f}")


# In[ ]:




