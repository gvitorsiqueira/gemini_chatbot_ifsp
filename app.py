__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import google.generativeai as genai
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from langchain.prompts import ChatPromptTemplate

# App title
st.set_page_config(page_title="💬 IFSP Chatbot")

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
O texto a seguir é um extrato do edital contendo informações sobre o processo seletivo dos cursos Técnicos e Técnicos integrados ao ensino médio do IFSP.  Estamos conversando sobre as vagas e o conteúdo programático dos cursos em diferentes campi.

{context}

---
É crucial que você encontre a resposta dentro do texto fornecido. Sua resposta deve ser baseada exclusivamente nas informações presentes no contexto acima.
Priorize informações de erratas ou retificações em relação a informações anteriores no contexto.
Responda a questão com base no contexto acima: {question}

**Exemplos de Perguntas Sinônimas:**

* O que vai cair na prova?
* Quais as matérias que devo estudar?
* Qual o conteúdo programático?
Outros sinônimos:
* Quais as vagas?
* Quantas vagas tem?

**Por favor, formate sua resposta usando Markdown.**

Histórico da Conversa:
{conversation_history}
"""

google_api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)
genai.GenerationConfig(temperature=1) 

# Variável global para o histórico da conversa
conversation_history = ""

def generate_response(prompt_input):
    global conversation_history

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = genai.GenerativeModel('gemini-1.5-flash')

    # Atualiza o histórico da conversa
    conversation_history += f"User: {prompt_input}\n"

    # Search the DB.
    results = db.similarity_search_with_score(prompt_input, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # Prepara o prompt com o histórico da conversa
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    input = prompt_template.format(context=context_text, question=prompt_input, conversation_history=conversation_history)
    # Gera a resposta
    output = model.generate_content(input)

    # Extrai a resposta
    full_response = ''
    for item in output._result.candidates[0].content.parts:
        full_response += item.text

    # Atualiza o histórico da conversa com a resposta
    conversation_history += f"Assistant: {full_response}\n"

    return full_response

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Eu sou o assistente virtual do processo seletivo IFSP. Como posso ajudar?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
