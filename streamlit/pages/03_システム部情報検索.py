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
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.agents import AgentExecutor
# from google.cloud import bigquery
# from langchain.agents import create_sql_agent
# from langchain.agents.agent_toolkits import SQLDatabaseToolkit
# from langchain.sql_database import SQLDatabase
# from sqlalchemy import *
# from sqlalchemy.engine import create_engine
# from sqlalchemy.schema import *
# from langchain.agents.agent_types import AgentType
# from langchain_experimental.agents.agent_toolkits import create_python_agent
# from langchain_experimental.tools import PythonREPLTool
import pandas as pd

st.set_page_config(
    page_title="システム部の情報検索",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("### システム部の情報検索")

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

hagging_model_name = "intfloat/multilingual-e5-large"

@st.cache_resource
def create_llm():
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0) #, max_tokens=4096)
    llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
    return llm


# インデックス化用のモデルダウンロード
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
# from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DataFrameLoader

# ##############################
# Index作成　Index変更時のみ
################################
# system_docs_list = []
# columns_to_read = ['instruction', 'output']
# out_word_file_name = r".\data\word_for_E5.csv"
# out_wiki_file_name = r".\data\wiki_for_E5.csv"
# out_knowledge_file_name = r".\data\knowledge_for_E5.csv"
# out_ticket_file_name = r".\data\ticket_for_E5.csv"

# out_csv_list = [out_word_file_name, out_wiki_file_name, out_knowledge_file_name, out_ticket_file_name]
# for out_csv in out_csv_list:
#     df = pd.read_csv(out_csv)
#     df['target'] = df['query'] + ' : ' + df['passage']
#     st.dataframe(df)
#     Loader = DataFrameLoader(df, page_content_column='target')
#     docs = Loader.load()
#     system_docs_list.extend(docs)

# # チャンクの分割
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=514,
#     chunk_overlap=20,
# )

# texts = text_splitter.split_documents(system_docs_list)

# # インデックスの作成
# index = FAISS.from_documents(
#     documents=texts,
#     embedding=HuggingFaceEmbeddings(model_name=hagging_model_name),
# )

# index.save_local("./system_vectorstore")
# ##############################
# Index作成　Index変更時のみ END
################################



# VectorIndexの読み込み
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores.utils import DistanceStrategy

@st.cache_resource
def create_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name = hagging_model_name)
    vectorstore = FAISS.load_local(
                    "./system_vectorstore", 
                    embedding,
                    distance_strategy = DistanceStrategy.MAX_INNER_PRODUCT)
    return vectorstore



# question = "印刷加工指示システムとは？"

# 類似度スコア付きの出力
# 類似度スコアが高いものから順番に取得してきます
user_msg = st.text_input("ここに検索内容入力してEnter")
if user_msg:
    result_df = pd.DataFrame()
    vectorstore = create_vectorstore()
    found_docs = vectorstore.similarity_search_with_score(user_msg, k=5)
    for doc, score in found_docs:
        metadata = ''
        for key, value in doc.metadata.items():
            metadata = metadata + str(value)
        doc_pd = pd.DataFrame({'Score': [score], 
                               'Content': [doc.page_content], 
                               'Metadata': [metadata],
                                })
        result_df = pd.concat([result_df, doc_pd])
        # st.write(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")

    result_df = result_df.reset_index(drop=True)
    result_df['Metadata'] = result_df['Metadata'].replace("", "", regex=True)
    result_df['Score'] = result_df['Score'].round(3).apply(lambda x: f"{x:.3f}")
    result_df = result_df.sort_values(by='Score', ascending=True)
    st.table(result_df)
    
# for i in range(len(found_docs)):
#     document, score = found_docs[i]
#     st.write(f"score: {score}, document: {document.page_content}")
    

# if 'history' not in st.session_state:
#     st.session_state['history'] = []

# if 'generated' not in st.session_state:
#     st.session_state['generated'] = ["産業資材事業部の売上データとコンタクト情報について聞いてみてください！"]

# if 'past' not in st.session_state:
#     st.session_state['past'] = [""]
# # This container will be used to display the chat history.
# response_container = st.container()
# # This container will be used to display the user's input and the response from the ChatOpenAI model.
# container = st.container()
# with container:
#     with st.form(key='my_form', clear_on_submit=True):
        
#         user_input = st.text_input("Input:", placeholder="質問を入力してください。", key='input')
#         submit_button = st.form_submit_button(label='Send')
        
#     if submit_button and user_input:
#         question = "salesもしくはcontact情報について"
#         question = question + user_input
#         question = question + " 回答の件数が5件以上の場合は、上位5件を教えて。日本語で回答してください。"
#         # question = question + "回答はmarkdownの表で表示して。"
#         output = run_query(question)
#         # st.write(output)
        
#         st.session_state['past'].append(user_input)
#         st.session_state['generated'].append(output)

# if st.session_state['generated']:
#     with response_container:
#         for i in range(len(st.session_state['generated'])):
#             if st.session_state['past'][i] != "":
#                 message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')#, avatar_style="initials", seed="You") #avatar_style="big-smile")
#             message(st.session_state["generated"][i], key=str(i))#avatar_style="initials", seed="ROB",) # avatar_style="thumbs")

# retriever = vectorstore.as_retriever()
# docs = retriever.invoke(question)
# for doc in docs:
#     st.write(f"document: {doc.page_content}")
