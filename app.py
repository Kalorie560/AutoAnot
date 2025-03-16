import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os
from scipy.stats import kurtosis  # クルトシス計算用

st.title("音声自動アノテーション＆データセット保存システム")

# 現在の作業ディレクトリを表示（保存場所の確認用）
st.write("現在の作業ディレクトリ:", os.getcwd())

# ユーザ入力：サンプリング周波数、収録時間、変化の閾値（％）
fs = st.number_input("サンプリング周波数 (Hz):", min_value=8000, max_value=96000, value=44100, step=1000)
duration = st.number_input("収録時間 (秒):", min_value=1, max_value=300, value=10, step=1)
threshold_percentage = st.number_input("変化の閾値（％）", min_value=0, max_value=100, value=20, step=1)
threshold_ratio = threshold_percentage / 100  # 例: 20%なら0.2

# 使用するメトリクスの選択
metric_choice = st.selectbox("使用するメトリクスを選択してください", options=["RMS", "クルトシス", "クレストファクタ"])

# 録音開始ボタン
if st.button("録音開始"):
    st.write("録音中...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # 録音終了待ち
    st.write("録音終了")
    
    num_segments = int(duration)
    segments = []         # 各1秒の波形データ
    metric_values = []    # 各セグメントの選択されたメトリクス値
    labels = []           # 自動付与するラベル ("OK" または "NG")
    
    for i in range(num_segments):
        start_idx = int(i * fs)
        end_idx = int((i + 1) * fs)
        segment = audio_data[start_idx:end_idx].flatten()
        segments.append(segment)
        
        # 選択されたメトリクスに応じて値を計算
        if metric_choice == "RMS":
            value = np.sqrt(np.mean(segment ** 2))
        elif metric_choice == "クルトシス":
            # fisher=False とすることで通常のクルトシスを得る
            value = kurtosis(segment, fisher=False)
        elif metric_choice == "クレストファクタ":
            rms_value = np.sqrt(np.mean(segment ** 2))
            value = np.max(np.abs(segment)) / rms_value if rms_value > 0 else 0
        metric_values.append(value)
    
    # 初回のフレームは無条件に "OK"
    labels.append("OK")
    for i in range(1, num_segments):
        prev_value = metric_values[i - 1]
        curr_value = metric_values[i]
        relative_change = abs(curr_value - prev_value) / prev_value if prev_value > 0 else 0
        if relative_change >= threshold_ratio:
            labels.append("NG")
        else:
            labels.append("OK")
    
    # セッションステートに保存（後でデータセット保存に利用）
    st.session_state['segments'] = segments
    st.session_state['labels'] = labels
    st.session_state['fs'] = fs
    st.session_state['metric_values'] = metric_values
    st.session_state['metric_choice'] = metric_choice

    # 全体の波形表示：灰色の全体波形に、1秒ごとに OK (緑) / NG (赤) で上書き
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, duration, int(duration * fs))
    ax.plot(t, audio_data, color='gray', alpha=0.5)
    
    for i, segment in enumerate(segments):
        start_time = i
        end_time = i + 1
        seg_t = np.linspace(start_time, end_time, len(segment))
        color = "green" if labels[i] == "OK" else "red"
        ax.plot(seg_t, segment, color=color, linewidth=2)
        ax.text((start_time + end_time) / 2, np.max(segment), labels[i],
                color=color, fontsize=12, ha='center')
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    # 各フレームのメトリクス値と自動ラベルを表示
    st.write("各フレームの", metric_choice, "値と自動ラベル:")
    for i, (val, label) in enumerate(zip(metric_values, labels)):
        st.write(f"{i+1}秒: {metric_choice} = {val:.4f}, ラベル = {label}")

# データセット保存ボタン
if st.button("データセット保存"):
    if all(key in st.session_state for key in ['segments', 'labels', 'fs', 'metric_choice']):
        segments_array = np.array(st.session_state['segments'])
        labels_array = np.array(st.session_state['labels'])
        save_path = "dataset.npz"  # 必要に応じて絶対パスに変更可
        np.savez(save_path,
                 waveforms=segments_array,
                 labels=labels_array,
                 fs=st.session_state['fs'],
                 metric=st.session_state['metric_choice'])
        st.success(f"データセットを保存しました ({save_path})")
        st.write("現在の作業ディレクトリ:", os.getcwd())
    else:
        st.error("保存するデータが見つかりません。まずは録音を実施してください。")
