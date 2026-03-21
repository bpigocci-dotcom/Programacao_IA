#importando todos os módulos necessários
import pandas as pd #ferramenta para criar e alterar dados em tabelas
import numpy as np #ferramenta de análise automática

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns 
import matplotlib.pyplot as plyplot
import joblib

#etapa 2:importar o nosso dicionário de dados (importação dataset)

try:
    print("carregando arquivo'churn-data.csv'...")
    df = pd.read_csv('churn-data.csv') #ler arquivo e criar uma tabela
    print(f"Sucesso,{len(df)}linhas importadas")
except FileNotFoundError:
    print("O arquivo não pode ser encontrado na pasta.")
    exit()
    
    #etapa 3: pre processamento de dados (preparar a ia para ser treinada)
    #passo 1: separar pergunta(x) da resposta (y)
    # (x) é tudo menos a colua cancelou, sãõ as "pistas" pro modelo
X=df.drop ("cancelou",axis=1)
#(y) apenas a coluna 'cancelou', é o que ueremos que o modelo preveja
y =df['cancelou']

#passo 2: dividir o trino do teste 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
#test_size=0.2 separa 20% da massa de dados para testar o modelo


#passo 3: normalizando (colocando tudo na mesma escala)
scaler = StandardScaler()

#fit rtransform do treino: IA calcula média e desvio padrão do treino
X_train_scaled = scaler.fit_transform(X_train)

#transform no teste; usamos a ´regua calculada no treino
X_test_scaled=scaler.transform(X_test)

#Etapa 4:treinar o modelo e realizar a previsão de dados
#criando o modelo
#n_estimators= 100, cria 100 árvores de decisões
modelo_churn = RandomForestClassifier(n_estimators=100, random_state=42)

#treinar/ajustar a IA
modelo_churn.fit(X_train_scaled, y_train)

#prever as respostas
previsoes = modelo_churn.predict(X_test_scaled)

#etapa 5: avaliação do modelo
print("Relatório de performance")
print(classification_report(y_test,previsoes))

#Etapa 6: Deploy -> salvar o trabalho
joblib.dump(modelo_churn,'modelo_churn_v1.pkl')

joblib.dump(scaler,'padronizador_v1.pkl')  
print("arquivos de ML foram exportados com sucesso")
  