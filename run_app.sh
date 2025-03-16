#!/bin/bash
# run_app.sh

# 仮想環境のアクティベート（必要な場合は以下のコメントアウトを外してください）
# source venv/bin/activate

# Streamlitアプリを起動（app.pyが同じディレクトリにある前提）
streamlit run app.py --server.address 192.168.249.1
