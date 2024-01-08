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
    page_title="産資売上コンタクト検索",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.write("### Demo of SQLDatabaseToolkit")

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

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

# SQLの例
few_shots = {
    "List all salesperson who registered contact data.": "SELECT DISTINCT salesperson_registered FROM contact;",
    "Find all company name of customer visited by '武藤 知典'.": "SELECT DISTINCT customer_company FROM contact WHERE salesperson_registered = '武藤 知典';",
    "List all branches.": "SELECT DISTINCT branch FROM contact;",
    "Find all company name of customer visited by salesperson at '東京支店' branch.": "SELECT DISTINCT customer_company FROM contact WHERE branch = '東京支店';",
    "List all salespersons who visite to a campany named '株式会社山善'.": "SELECT DISTINCT salesperson_registered FROM contact WHERE customer_company = '株式会社山善';",
    "How many companys are there visited by '武藤 知典'?": "SELECT count(*) AS companys FROM (SELECT customer_company FROM contact WHERE salesperson_registered = '武藤 知典' GROUP BY customer_company) a;",
    "Find the total number of companys which ware visited by salespersons at '名古屋支店' branch for promoting '折りコン'.": "SELECT count(*) AS companys FROM (SELECT customer_company FROM contact WHERE branch = '名古屋支店' AND contact_details LIKE '%#折りコン%' GROUP BY customer_company) a;",
    "List all details of contacts and company name which '五十住 友彰' made.": "SELECT customer_company, contact_details FROM contact WHERE salesperson_registered = '五十住 友彰';",
    "Who are the top 5 salesperson at '東京支店' branch by total sales contacts?": "SELECT TOP(5) salesperson_registered, COUNT(*) AS sales_contacts FROM contact WHERE branch = '東京支店' GROUP BY salesperson_registered ORDER BY sales_contacts DESC;",
    "List number og contact made by '五十住 友彰' each month in 2023?": "SELECT contact_month, count(*) AS number_of_contacts FROM (SELECT MONTH(contact_date) AS contact_month FROM contact WHERE salesperson_registered = '五十住 友彰' AND YEAR(contact_date) = 2023) a  GROUP BY contact_month;",
    "How many annual sales contacts in August did each salesperson make ?": "SELECT salesperson_registered, YEAR(contact_date), COUNT(salesperson_registered) AS CONTACT_TIMES FROM contact WHERE MONTH(contact_date) = 8 GROUP BY salesperson_registered, YEAR(contact_date) ORDER BY CONTACT_TIMES DESC;",
    "What is the total sales amount of the department of 'TA' in August 2023?": "SELECT department, SUM(amount) AS total_sales_amount FROM sales WHERE department = 'TA' AND MONTH(sales_date) = 8 AND YEAR(sales_date) = 2023 GROUP BY department;",
    "What are the top 5 products that sold the most in September 2023?": "SELECT TOP(5) s.product_code,p.product_name,count(s.product_code) AS number_of_sold FROM sales s LEFT JOIN product_master p ON s.product_code = p.product_code WHERE MONTH(sales_date) = 9 AND YEAR(sales_date) = 2023 AND s.product_code NOT LIKE '%ZZ%' GROUP BY s.product_code,p.product_name ORDER BY number_of_sold desc;",
    "What is sales amount of '五十住 友彰' for each agency in May 2023?": "SELECT  .agency_code ,a.abbreviation ,SUM(s.amount) AS total_sales_amount FROM sales s JOIN agency_master a ON s.agency_code = a.agency_code JOIN employee_master e ON a.employee_code = e.employee_code WHERE e.employee_name = '五十住 友彰' AND MONTH(s.sales_date) = 5 GROUP BY  s.agency_code ,a.abbreviation ORDER BY total_sales_amount DESC;SELECT  s.agency_code ,a.abbreviation ,SUM(s.amount) AS total_sales_amount FROM sales s JOIN agency_master a ON s.agency_code = a.agency_code JOIN employee_master e ON a.employee_code = e.employee_code WHERE e.employee_name = '五十住 友彰' AND MONTH(s.sales_date) = 5 and YEAR(s.sales_date) = 2023 GROUP BY  s.agency_code ,a.abbreviation ORDER BY total_sales_amount DESC;",
    "List top 3 salespersons and their sales amount in '東京支店' branch.": "SELECT TOP(3) e.employee_name, sum(s.amount) FROM sales s LEFT JOIN agency_master a ON s.agency_code = a.agency_code LEFT JOIN employee_master e ON a.employee_code = e.employee_code WHERE s.branch = '東京支店' AND e.employee_code IS NOT NULL GROUP BY e.employee_name ORDER BY sum(s.amount) DESC;",
    "What is anual seles amount of the '名古屋支店'?": "SELECT YEAR(sales_date), sum(s.amount) FROM sales s WHERE s.branch = '名古屋支店' GROUP BY YEAR(s.sales_date) ORDER BY YEAR(s.sales_date) ASC;",
}

# インデックス化用のモデルダウンロード
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory

@st.cache_resource#(allow_output_mutation=True)
def create_embedding():
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    return embedding

# インデックス作成
few_shot_docs = [
    Document(page_content=question, metadata={"sql_query": few_shots[question]})
    for question in few_shots.keys()
]

# チャンクの分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=514,
    chunk_overlap=20,
)

texts = text_splitter.split_documents(few_shot_docs)

# インデックスの作成
index = FAISS.from_documents(
    documents=texts,
    embedding=create_embedding(),
)

# retriverの作成
retriever = index.as_retriever()

tool_description = """
This tool will help you understand similar examples to adapt them to the user question.
Input to this tool should be the user question.
"""

retriever_tool = create_retriever_tool(
    retriever, name="sql_get_similar_examples", description=tool_description
)
custom_tool_list = [retriever_tool]

custom_suffix = """
I should first get the similar examples I know.
If the examples are enough to construct the query, I can build it.
Otherwise, I can then look at the tables in the database to see what I can query.
Then I should query the schema of the most relevant tables
"""

# SQL Serverへの接続設定
server = 'SO-PC'
database = 'datamart'
username = 'gpkadmin'
password = '19vK8xEQ'
connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server'

# SQLAlchemyエンジンを作成
db = SQLDatabase.from_uri(connection_string)

# Vertex AI 基盤モデルを初期化 OPEN AI利用
llm = create_llm()

# SQL データベースと対話するエージェントの初期化
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    # agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    # top_k=10,
    extra_tools=custom_tool_list,
    suffix=custom_suffix,
    memory=ConversationBufferMemory(return_messages=True)
)
# サイドバーにリロードボタンをつける
st.sidebar.button('Reload')

def run_query(query):
    res = agent_executor.run(query)
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
        for chunk in response:
            if chunk.choices[0].finish_reason is not None:
                break
            # 回答を逐次表示
            assistant_msg += chunk.choices[0].delta.content
            assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})
