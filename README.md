# Readme
## 環境構築
### gitからクローン
1. 適当なフォルダでgit clone
   ```
   git clone https://XXXXXXXXXXXX
   ```
### pythonの仮想環境作成
1. 仮想環境作成
   <br>
   下記コマンドでstreamlitというフォルダができて、配下に仮想環境が作成される。
   ``` 
   python -m venv streamlit
   ```
   pytnonのバージョンを指定する場合はこっち（下記は3.10）
   ```
   py -3.10 -m venv streamlit
   ```
2. 仮想環境の起動
   <br>
   下記コマンドで仮想環境起動。ターミナルの先頭に仮想環境名が（）で括って表示される。
   ```
   cd streamlit
   Scripts\activate
   ```

   スクリプトの実行ができないとでたら、PowerShellで下記実行（ポリシー変更しないとデフォルトではpsファイルが実行できない）
   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
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
2. OpenApiのキーを登録
.envファイルを作成し下記を記載
   ```
   OPENAI_API_KEY="OpenApiのキー"
   ```
3. 実行
   ```
   streamlit run Top.py
   ```