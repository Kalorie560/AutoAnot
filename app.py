import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os

st.title("音声自動アノテーション＆データセット保存システム")

# 現在の作業ディレクトリを表示（保存場所の確認用）
st.write("現在の作業ディレクトリ:", os.getcwd())

# ユーザ入力：サンプリング周波数、収録時間、RMS変化の閾値（％）
fs = st.number_input("サンプリング周波数 (Hz):", min_value=8000, max_value=96000, value=44100, step=1000)
duration = st.number_input("収録時間 (秒):", min_value=1, max_value=300, value=10, step=1)
threshold_percentage = st.number_input("RMS変化の閾値（％）", min_value=0, max_value=100, value=20, step=1)
threshold_ratio = threshold_percentage / 100  # 例: 20%なら0.2となる

# 録音開始ボタン
if st.button("録音開始"):
    st.write("録音中...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # 録音終了待ち
    st.write("録音終了")
    
    # 1秒ごとに音声を分割し、RMS値の計算とラベル付与を実施
    num_segments = int(duration)
    segments = []   # 各1秒の波形データを格納
    rms_values = [] # 各セグメントのRMS値
    labels = []     # 自動付与するラベル（"OK"または"NG"）
    
    for i in range(num_segments):
        start_idx = int(i * fs)
        end_idx = int((i + 1) * fs)
        segment = audio_data[start_idx:end_idx].flatten()  # 1次元配列に変換
        segments.append(segment)
        rms = np.sqrt(np.mean(segment ** 2))
        rms_values.append(rms)
    
    # 初回のフレームは無条件に "OK"
    labels.append("OK")
    for i in range(1, num_segments):
        prev_rms = rms_values[i - 1]
        curr_rms = rms_values[i]
        relative_change = abs(curr_rms - prev_rms) / prev_rms if prev_rms > 0 else 0
        if relative_change >= threshold_ratio:
            labels.append("NG")
        else:
            labels.append("OK")
    
    # セッションステートに結果を保存（後から保存ボタンで利用）
    st.session_state['segments'] = segments
    st.session_state['labels'] = labels
    st.session_state['fs'] = fs

    # 全体の波形表示：灰色で全体波形、1秒ごとに色分け（OK:緑、NG:赤）
    fig, ax = plt.subplots(figsize=(10, 4))
    t = np.linspace(0, duration, int(duration * fs))
    ax.plot(t, audio_data, color='gray', alpha=0.5)
    
    for i, segment in enumerate(segments):
        start_time = i
        end_time = i + 1
        seg_t = np.linspace(start_time, end_time, len(segment))
        color = "green" if labels[i] == "OK" else "red"
        ax.plot(seg_t, segment, color=color, linewidth=2)
        ax.text((start_time + end_time) / 2, np.max(segment), labels[i], color=color,
                fontsize=12, ha='center')
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)
    
    # 各フレームのRMS値とラベルの一覧表示
    st.write("各フレームのRMS値と自動ラベル:")
    for i, (r, label) in enumerate(zip(rms_values, labels)):
        st.write(f"{i+1}秒: RMS={r:.4f}, ラベル={label}")

# データセット保存ボタン（録音後にセッションステートにデータがあれば動作）
if st.button("データセット保存"):
    if 'segments' in st.session_state and 'labels' in st.session_state and 'fs' in st.session_state:
        segments_array = np.array(st.session_state['segments'])  # shape: (num_segments, サンプル数)
        labels_array = np.array(st.session_state['labels'])
        save_path = "dataset.npz"  # 必要に応じて絶対パスに変更可能
        np.savez(save_path, waveforms=segments_array, labels=labels_array, fs=st.session_state['fs'])
        st.success(f"データセットを保存しました ({save_path})")
        st.write("現在の作業ディレクトリ:", os.getcwd())
    else:
        st.error("保存するデータが見つかりません。")
