import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="TOP Page",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# This is a header."
    }
)

st.title('LLMテスト　トップページ')
st.write('### サイドバーから機能を選択してください')