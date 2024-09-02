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
O texto a seguir é um extrato do edital contendo informações sobre o processo seletivo dos cursos Técnicos e Técnicos integrados ao ensino médio do IFSP:

{context}

---
É crucial que você encontre a resposta dentro do texto fornecido. Sua resposta deve ser baseada exclusivamente nas informações presentes no contexto acima.
Priorize informações de erratas ou retificações em relação a informações anteriores no contexto.
Responda a questão com base no contexto acima: {question}

**Por favor, formate sua resposta usando Markdown.**
"""
google_api_key=st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=google_api_key)
genai.GenerationConfig(temperature=0.5)

def generate_response(input_text):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response_text = model.generate_content(input_text)
    texto_resposta = response_text._result.candidates[0].content.parts[0].text
    st.info(texto_resposta)



# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Olá! Eu sou o assistente virtual do processo seletivo IFSP. Como posso ajudar?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Como eu posso ajudar?"}]


# Function for generating Gemini response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_response(prompt_input):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    model = genai.GenerativeModel('gemini-1.5-flash')
    string_dialogue = "Você é um assistente pessoal. Você não responde como 'User' ou finge ser 'User'. Você apenas responde como 'Assistant'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"

    # Search the DB.
    results = db.similarity_search_with_score(prompt_input, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    input = prompt_template.format(context=context_text, question=prompt_input)
    #input=f"{string_dialogue} {prompt_input} Assistant:"
    output = model.generate_content(input)
    return output

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
            full_response = ''
            for item in response._result.candidates[0].content.parts:
                full_response += item.text
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)

#st.button('Clear Chat History', on_click=clear_chat_history)
