import os
import streamlit as st
from streamlit_chat import message
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
# import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.agents import AgentExecutor
# from google.cloud import bigquery
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
# from sqlalchemy import *
# from sqlalchemy.engine import create_engine
# from sqlalchemy.schema import *
from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_python_agent
# from langchain_experimental.tools import PythonREPLTool

st.set_page_config(
    page_title="産資コンタクト検索",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("### 産資　コンタクト要約など")
st.write("東京支店の９月の訪問内容を２００文字で要約して。など。")
st.write("コンタクト情報をVector情報に変換後、OpenAIのAPIを利用して内容要約しています。")

# os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

@st.cache_resource
def create_llm():
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0) #, max_tokens=4096)
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    return llm

# agent_executor = create_python_agent(
#     llm=llm,
#     tool=PythonREPLTool(),
#     verbose=True,
#     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
# )

# answer = agent_executor.run(
# """
# ランダムなデータでDataframeを作って
# """
# )

# st.write(answer)



# インデックス化用のモデルダウンロード
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory


# VectorIndexの読み込み
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

hagging_model_name="intfloat/multilingual-e5-large"
embedding = HuggingFaceEmbeddings(model_name = hagging_model_name)

# load
@st.cache_resource
def create_vectorstore():
    vectorstore = FAISS.load_local("./vectorstore", embedding)
    return vectorstore


from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# prompt_template_qa = """あなたは親切で優しいアシスタントです。丁寧に、日本語でお答えください！
# もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問
# に答えてください。
prompt_template_qa = """あなたは営業マンです。簡潔に、事実に基づいて、日本語で回答してください。
もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問
に答えてください。
{context}

質問: {question}
回答（日本語）:"""

prompt_qa = PromptTemplate(
        template=prompt_template_qa,
        input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": prompt_qa}

retriever=create_vectorstore().as_retriever(
        search_kwargs={"k": 35}, # Loraモデルは5~7,gpt3.5-16kは40、gpt4は400
        search_type="similarity",
        )

# 質問応答チェーンの作成
qa_chain = RetrievalQA.from_chain_type(
    llm=create_llm(),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs,
    # return_source_documents=True
)

# サイドバーにリロードボタンをつける
st.sidebar.button('Reload')

def run_query(query):
    res = qa_chain.run(query)
    return res

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_msg = st.chat_input("ここにメッセージを入力")

if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_msg)

    # アシスタントのメッセージを表示
    response = run_query(user_msg)
    with st.chat_message(ASSISTANT_NAME):
        assistant_msg = ""
        assistant_response_area = st.empty()
        # for chunk in response:
        #     if chunk.choices[0].finish_reason is not None:
        #         break
            # 回答を逐次表示
        assistant_msg += response
        assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})
