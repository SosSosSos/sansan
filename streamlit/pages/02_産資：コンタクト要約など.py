import os
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# 定数
EMBEDDING_MODEL_NAME="intfloat/multilingual-e5-large"
CHAT_MOCEL_NAME = "gpt-4-1106-preview"
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# ロジック
# LLMのロード
@st.cache_resource
def create_llm():
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0) #, max_tokens=4096)
    llm = ChatOpenAI(model_name=CHAT_MOCEL_NAME, temperature=0)
    return llm

# Vector用モデルのロード
@st.cache_resource
def create_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name = EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local("./vectorstore", embedding)
    return vectorstore


# プロンプト生成
def create_prompt():
    prompt_template_qa = """あなたは営業マンです。簡潔に、事実に基づいて、日本語で回答してください。
    もし以下の情報が探している情報に関連していない場合は、そのトピックに関する自身の知識を用いて質問
    に答えてください。
    {context}

    質問: {question}
    回答（日本語）:"""

    prompt_qa = PromptTemplate(
            template=prompt_template_qa,
            input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt_qa}
    return chain_type_kwargs

# 質問応答チェーンの作成
def create_chain():
    # リトライバーの作成
    retriever=create_vectorstore().as_retriever(
            search_kwargs={"k": 300}, # Loraモデルは5~7,gpt3.5-16kは40、gpt4は400
            search_type="similarity",
            )
    qa_chain = RetrievalQA.from_chain_type(
        llm=create_llm(),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=create_prompt(),
        # return_source_documents=True
        )
    return qa_chain

def run_query(query):
    res = create_chain().run(query)
    return res

# ページ描画
st.set_page_config(
    page_title="産資コンタクト検索",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)
# リロードボタン
st.sidebar.button('Reload')

# ページ内容
st.write("### 産資　コンタクト要約など")
st.write("東京支店の９月の訪問内容を２００文字で要約して。など。")
st.write("コンタクト情報をVector情報に変換後、OpenAIのAPIを利用して内容要約しています。")

# チャットログを保存したセッション情報を初期化
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

user_msg = st.chat_input("ここに質問を入力")

if user_msg:
    # 以前のチャットログを表示
    for chat in st.session_state.chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # 最新のメッセージを表示
    with st.chat_message(USER_NAME):
        st.write(user_msg)

    # Chain実行、アシスタントのメッセージを表示
    try:
        response = run_query(user_msg)
    except Exception as e:
        response = f"エラーが発生しました: {e}"

    with st.chat_message(ASSISTANT_NAME):
        assistant_msg = ""
        assistant_response_area = st.empty()

        # 回答を逐次表示
        assistant_msg += response
        assistant_response_area.write(assistant_msg)

    # セッションにチャットログを追加
    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})
