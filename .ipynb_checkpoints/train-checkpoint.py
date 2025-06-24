#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# In[9]:


bucket_path = 'gs://predictive-maintenance-leopoldo/manutpred.csv'
df_raw = pd.read_csv(bucket_path, storage_options={'token': 'cloud'})


# In[10]:


df = df_raw.copy()
df.head()


# In[11]:


drop_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
df.drop(drop_columns, axis=1, inplace=True)


# In[12]:


df.head()


# In[13]:


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


# In[15]:


X_train.head()


# In[17]:


y_test.head()


# In[32]:


# CATEGORICAL --> criando o encoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# encoder + df de treino
encoded_type_train = encoder.fit_transform(X_train[categorical_columns])
encoded_train_df = pd.DataFrame(
    encoded_type_train,
    columns=encoder.get_feature_names_out(input_features=categorical_columns),
    index=X_train.index
)


# In[34]:


# CATEGORICAL --> encoder + df de teste
encoded_type_test = encoder.transform(X_test[categorical_columns])

encoded_test_df = pd.DataFrame(
    encoded_type_test,
    columns=encoder.get_feature_names_out(input_features=categorical_columns),
    index=X_test.index
)


# In[43]:


# NUMERICAL --> Scaler
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_columns])
X_test_num = scaler.transform(X_test[numerical_columns])


# In[44]:


X_train_processed = pd.DataFrame(
    np.hstack([X_train_num, encoded_train_df]),
    columns=numerical_columns + list(encoder.get_feature_names_out(categorical_columns)),
    index=X_train.index
)


# In[45]:


X_test_processed = pd.DataFrame(
    np.hstack([X_test_num, encoded_test_df]),
    columns=numerical_columns + list(encoder.get_feature_names_out(categorical_columns)),
    index=X_test.index
)


# In[47]:


X_train_processed.head()


# In[48]:


X_test_processed.head()


# In[50]:


model = RandomForestClassifier(random_state=42)
model.fit(X_train_processed, y_train)

preds = model.predict(X_test_processed)

print(classification_report(y_test, preds))


# In[ ]:




