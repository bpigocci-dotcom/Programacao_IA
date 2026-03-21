#importando as libs necessárias
import pandas as pd 
import numpy as np 

#criando numeros aleatórios pra simular dados reais
#definindo uma semente para fins de simulação
np.random.seed(42)

#gerando 500 registros 
n_registros = 500
#estruturando os dados do arquivo .csv
data = {
    
'tempo_contrato': np.random.randint(1,48,n_registros), #1 a 48 meses
'valor_mensal': np.random.uniform(50.0,150.0,n_registros).round(2),
#assinatura com valores que variam de 50 a 150 dinheiros
'reclamacoes' : np.random.poisson(1.5,n_registros)
#cada user tem uma média de 1.5 reclamaçôes
}

#convertendo a estrutura de dicionário em um conjunto de dados
df = pd.DataFrame(data)

#criar asimulação da lógica de churn
#o cliente tem mais chances de sair se tiver muitas reclamações ou 
#se o contrato for curto
df['cancelou']=((df['reclamacoes']>2|(df['tempo_contrato']<6))).astype(int)

#salvando o dataset em .csv
df.to_csv('churn-data.csv',index=False)
print("arquivo 'churn_data.csv'gerado com sucsso!")



