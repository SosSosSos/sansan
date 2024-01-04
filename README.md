# Readme
## 環境構築
### pythonの仮想環境作成
1. 仮想環境作成
   <br>
   下記コマンドでstreamlitというフォルダができて、配下に仮想環境が作成される。
   ``` 
   python -m venv streamlit
   ```
2. 仮想環境の起動
   <br>
   下記コマンドで仮想環境起動。ターミナルの先頭に仮想環境名が（）で括って表示される。
   ```
   cd streamlit
   Scripts\activate
   ```
3. 仮想環境のpipのアップデート
   ```
   python.exe -m pip install --upgrade pip
   ```
### streamlitのインストール
1. pipでインストール
```
pip install streamlit
```
2. 動作確認（サンプルプロジェクト）
```
streamllit hello
```

### LLM動作環境の作成
1. ライブラリインストール
   ```
   pip install streamlit_chat langchain openai faiss-cpu tiktoken pypdf sqlalchemy google-cloud-bigquery langchain_experimental sentence-transformers pyodbc
   ```
   下記はLanChainのデバッグ用
   ```
   pip install -U wandb
   ```
2. 実行
   ```
   streamlit run langChainExp.py
   ```
3. 