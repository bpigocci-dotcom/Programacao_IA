#importar módulos
import streamlit as st #lib que transforma python em sites web
import joblib 
import numpy as np

#passo 1: configurando a aba do navegador
st.set_page_config(page_title='análise de churn',page_icon="🫥")
#textos da tela principal
st.title("Sistema de retenção de base") #titulo da pagina
st.markdown('insira dados do cliente para verificar risco de cancelamento')

#passo 2: Importaros dados da inteligencia artificial com o joblib
modelo = joblib.load('modelo_churn_v1.pkl') #carrega as regras de decisão do modelo
scaler = joblib.load('padronizador_v1.pkl') #carrega a régua matemática

#passo 3: Criar a interfac de entrada com formulário
col1, col2  = st.columns(2) #criando duas colunas 

#coluna lado esquerdo (col1)
with col1:
    tempo = st.number_input("Tempo de contrato (meses)", min_value=1, value=12, max_value=200)
    valor = st.number_input("valor da assinatura: (R$)" , min_value=0.0, value=50.0)

with col2:
    reclamacoes = st.slider("Histórico de Reclamações", 0,10,1)
    
#passo 4: processamento de dados
if st.button("Analisar Risco"):
    dados = scaler.transform([[tempo, valor, reclamacoes]])
    probabilidade = modelo.predict_proba(dados)[0][1]#previsão de probabilidade  
 

#passo 5: feedback de negócios
    st.divider()#cria uma linha

#probabilidade maior <70%
    if probabilidade >0.7:
        st.error(f"*ALTO RISCO DE CHURN* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:*Oferecer cupom de fidelidade:FID210360OFF")
        
    elif probabilidade >0.3:
        st.warning(f"*Risco moderado de Churn* ({probabilidade*100:.1f}%)")
        st.info("*Sugestão de ação:*Realizar chamada de acompanhamento.")
        
    else:
        st.success(f"*Cliente estável* ({probabilidade*100:.1f}%)")
        st.info("Nada a Realizar no momento.")
    


