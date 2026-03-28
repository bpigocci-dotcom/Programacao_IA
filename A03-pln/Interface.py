import streamlit as st
import joblib
import spacy
import pandas as pd

# configuração da página
st.set_page_config(page_title="Triagem de chamados", page_icon="🐙")

# -------------------------------
# Carregamento dos recursos
# -------------------------------

@st.cache_resource
def carregar_modelo():
    return joblib.load("modelo_triagem_suporte.pkl")

@st.cache_resource
def carregar_nlp():
    nlp_model = spacy.load("pt_core_news_sm")
    
    if nlp_model is None:
        raise ValueError("Erro ao carregar o modelo NLP")
    
    return nlp_model

# tentativa de carregar os recursos
try:
    modelo = carregar_modelo()
    nlp = carregar_nlp()
    
except Exception as e:
    st.error("Erro ao carregar recursos. Execute o script de treinamento para gerar o modelo.")
    st.error(str(e))
    st.stop()

# -------------------------------
# Lógica de processamento
# -------------------------------

def analisar_chamado(texto_usuario):
    
    doc = nlp(texto_usuario)
    
    # entidades nomeadas
    entidades = [(ent.text, ent.label_) for ent in doc.ents]
    
    # limpeza do texto (com remoção de stopwords)
    texto_limpo = " ".join([
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_stop
    ])
    
    # predição
    categoria_predita = modelo.predict([texto_limpo])[0]
    
    probs = modelo.predict_proba([texto_limpo])[0]
    confianca = max(probs) * 100
    
    return categoria_predita, confianca, entidades

# -------------------------------
# Interface gráfica
# -------------------------------

st.title("Triagem de suporte")
st.markdown("Descreva o problema em poucas palavras.")

# histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# exibir mensagens anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# entrada do usuário
if prompt := st.chat_input("Ex.: O servidor AWS parou de responder..."):
    
    if not prompt.strip():
        st.warning("Digite uma descrição do problema.")
        st.stop()

    # exibir mensagem do usuário
    st.chat_message("user").markdown(prompt)
    
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # análise
    categoria, confianca, ents = analisar_chamado(prompt)

    # ações automáticas
    acoes = {
        "Infraestrutura": "Encaminhando para equipe N2",
        "Acesso": "Verificando logs de autenticação.",
        "Hardware": "Abrindo ordem de serviço.",
        "Software": "Verificando disponibilidade de licenças"
    }

    # resposta formatada
    resposta_md = f"""
**Análise do chamado:**
- **Categoria:** `{categoria}`
- **Confiança:** `{confianca:.2f}%`
"""

    if ents:
        resposta_md += "\n\n**Entidades detectadas:**"
        for ent in ents:
            resposta_md += f"\n- *{ent[0]}* ({ent[1]})"

    resposta_md += f"\n\n**Ação sugerida:** {acoes.get(categoria, 'Triagem manual necessária.')}"

    # exibir resposta
    with st.chat_message("assistant"):
        st.markdown(resposta_md)

    # salvar no histórico
    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta_md
    })