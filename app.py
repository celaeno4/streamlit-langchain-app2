import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder

import os
import openai
import dotenv
dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")

def create_agent_chain():
    chat = ChatOpenAI(
        model_name = "gpt-4",
        temperature=0.5,
        streaming=True,
    )

    # OpenAI Functions Agent のプロンプトに Memory の会話履歴を追加するための設定
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }

    # OpenAI Functions Agent が仕える設定で Memory を初期化
    memory = ConversationBufferMemory(
        memory_key="memory",
        return_messages=True,
    )

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

st.title("langchain-streamlit-app")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    # ユーザの入力内容を st.settion_state.messages に追加
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        #agent_chain = create_agent_chain()
        #response = agent_chain.run(prompt, callbacks=[callback])
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])
        st.markdown(response)

        #chat = ChatOpenAI(
        #    model_name="gpt-4",
        #    temperature=0.5,
        #    )
        #messages = [HumanMessage(content=prompt)]
        #response = chat(messages)
        #st.markdown(response.content)

    #with st.chat_message("assistant"):
    #    response = "こんにちは"
    #    st.markdown(response)

    # 応答を st.settion_state.messages に追加
    st.session_state.messages.append({"role": "assistant", "content": response})