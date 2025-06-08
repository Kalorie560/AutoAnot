import streamlit as st
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import os
from scipy.stats import kurtosis  # クルトシス計算用

st.title("🎙️ 音声自動アノテーション＆データセット保存システム")

# 使用方法の説明
st.markdown("""
### 📖 使用方法
1. **設定**: サンプリング周波数、録音時間、変化閾値、メトリクスを設定
2. **録音**: 「録音開始」ボタンで音声を録音
3. **確認**: 自動生成された波形とラベルを確認
4. **編集**: 必要に応じてOK/NGラベルを手動で変更
5. **保存**: 「データセット保存」でdataset.npzファイルとして保存

---
""")

# 使用方法の説明
st.markdown("""
### 📖 使用方法
1. **設定**: サンプリング周波数、収録時間、変化閾値を設定
2. **録音**: 「録音開始」ボタンで音声を収録
3. **自動分析**: システムが自動的に各セグメントにOK/NGラベルを付与
4. **手動編集**: 必要に応じて各セグメントのラベルを手動で変更
5. **波形確認**: 「波形を更新」ボタンで編集結果を視覚的に確認
6. **保存**: 「データセット保存」ボタンで最終データセットを保存

---
""")

# 現在の作業ディレクトリを表示（保存場所の確認用）
st.write("📁 **現在の作業ディレクトリ**:", os.getcwd())

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
    
    # 手動編集用のラベルを初期化（自動ラベルをコピー）
    if 'manual_labels' not in st.session_state:
        st.session_state['manual_labels'] = labels.copy()

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
    
    # ラベル編集用のセッションステート初期化
    if 'editable_labels' not in st.session_state:
        st.session_state['editable_labels'] = labels.copy()
    else:
        # 新しい録音データの場合は更新
        st.session_state['editable_labels'] = labels.copy()
    
    # ラベル分布表示
    st.write("### 📊 ラベル分布")
    auto_ok = labels.count("OK")
    auto_ng = labels.count("NG")
    manual_ok = st.session_state['editable_labels'].count("OK")
    manual_ng = st.session_state['editable_labels'].count("NG")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("自動ラベル", f"OK: {auto_ok}, NG: {auto_ng}")
    with col2:
        st.metric("手動ラベル", f"OK: {manual_ok}, NG: {manual_ng}")
    
    # ラベルリセットボタン
    if st.button("🔄 自動ラベルに戻す"):
        st.session_state['editable_labels'] = labels.copy()
        st.rerun()
    
    # 各フレームのメトリクス値と自動ラベルを表示

    st.write("各フレームの", metric_choice, "値と自動ラベル:")
    for i, (val, label) in enumerate(zip(metric_values, labels)):
        st.write(f"{i+1}秒: {metric_choice} = {val:.4f}, ラベル = {label}")
    
    st.markdown("---")

# 🎛️ ラベル手動編集セクション
if 'segments' in st.session_state and 'labels' in st.session_state:
    st.markdown("### 🎛️ ラベル手動編集")
    st.write("各時間セクションのラベルを手動で変更できます。変更後は「波形を更新」ボタンで結果を確認してください。")
    
    # 手動編集用のラベルを初期化（まだ存在しない場合）
    if 'manual_labels' not in st.session_state:
        st.session_state['manual_labels'] = st.session_state['labels'].copy()
    
    # ラベル分布の表示
    col1, col2 = st.columns(2)
    with col1:
        auto_ok = st.session_state['labels'].count('OK')
        auto_ng = st.session_state['labels'].count('NG')
        st.metric("自動ラベル", f"OK: {auto_ok}, NG: {auto_ng}")
    
    with col2:
        manual_ok = st.session_state['manual_labels'].count('OK')
        manual_ng = st.session_state['manual_labels'].count('NG')
        st.metric("手動編集後", f"OK: {manual_ok}, NG: {manual_ng}")
    
    # 各セグメントのラベル編集
    st.write("**各セグメントのラベル選択:**")
    cols = st.columns(min(len(st.session_state['labels']), 5))  # 最大5列で表示
    
    for i, label in enumerate(st.session_state['labels']):
        col_idx = i % 5
        with cols[col_idx]:
            # ユニークなキーを使用してSelectboxを作成
            new_label = st.selectbox(
                f"秒{i+1}",
                options=["OK", "NG"],
                index=0 if st.session_state['manual_labels'][i] == "OK" else 1,
                key=f"label_edit_{i}"  # ユニークなキー
            )
            st.session_state['manual_labels'][i] = new_label
            
            # 変更されたラベルを視覚的に表示
            if st.session_state['manual_labels'][i] != st.session_state['labels'][i]:
                st.caption(f"🔄 変更: {st.session_state['labels'][i]} → {new_label}")
    
    # リセットボタン
    if st.button("🔄 自動ラベルに戻す", key="reset_labels_button"):
        st.session_state['manual_labels'] = st.session_state['labels'].copy()
        st.rerun()
    
    # 波形更新ボタン（ユニークなキーを追加）
    if st.button("📊 編集後のラベルで波形を更新", key="update_waveform_button"):
        # 手動編集されたラベルで波形を再描画
        duration = len(st.session_state['segments'])
        fs = st.session_state['fs']
        
        # 全体波形の再構築
        all_segments = np.concatenate(st.session_state['segments'])
        
        fig, ax = plt.subplots(figsize=(12, 5))
        t = np.linspace(0, duration, len(all_segments))
        ax.plot(t, all_segments, color='gray', alpha=0.4, label='元の音声')
        
        # 手動編集されたラベルで色分け表示
        for i, segment in enumerate(st.session_state['segments']):
            start_time = i
            end_time = i + 1
            seg_t = np.linspace(start_time, end_time, len(segment))
            
            # 手動編集されたラベルを使用
            manual_label = st.session_state['manual_labels'][i]
            auto_label = st.session_state['labels'][i]
            
            # ラベルが変更されたかどうかで表示を変える
            if manual_label != auto_label:
                # 変更されたセグメント：太い線で表示
                color = "darkgreen" if manual_label == "OK" else "darkred"
                ax.plot(seg_t, segment, color=color, linewidth=3, alpha=0.8)
                ax.text((start_time + end_time) / 2, np.max(segment) * 1.1, 
                       f"{manual_label}*", color=color, fontsize=12, ha='center', weight='bold')
            else:
                # 変更されていないセグメント：通常の表示
                color = "green" if manual_label == "OK" else "red"
                ax.plot(seg_t, segment, color=color, linewidth=2, alpha=0.7)
                ax.text((start_time + end_time) / 2, np.max(segment) * 1.1, 
                       manual_label, color=color, fontsize=10, ha='center')
        
        ax.set_xlabel("時間 (秒)")
        ax.set_ylabel("振幅")
        ax.set_title("編集後のアノテーション結果 (* = 手動編集)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # 編集サマリーの表示
        changed_count = sum(1 for i in range(len(st.session_state['labels'])) 
                          if st.session_state['manual_labels'][i] != st.session_state['labels'][i])
        if changed_count > 0:
            st.success(f"✅ 波形を更新しました。{changed_count}個のラベルが手動編集されています。")
        else:
            st.info("ℹ️ 波形を更新しました。手動編集はありません。")
    
    st.markdown("---")

# 💾 データセット保存セクション
st.markdown("### 💾 データセット保存")
if st.button("💾 データセット保存", key="save_dataset_button"):
    if all(key in st.session_state for key in ['segments', 'labels', 'fs', 'metric_choice']):
        segments_array = np.array(st.session_state['segments'])
        
        # 手動編集されたラベルがあればそれを使用、なければ自動ラベルを使用
        if 'manual_labels' in st.session_state:
            labels_to_save = st.session_state['manual_labels']
            st.info("📝 手動編集されたラベルを使用してデータセットを保存します。")
        else:
            labels_to_save = st.session_state['labels']
            st.info("🤖 自動生成されたラベルを使用してデータセットを保存します。")
        
        labels_array = np.array(labels_to_save)
        save_path = "dataset.npz"  # 必要に応じて絶対パスに変更可
        
        # 保存前の内容確認
        ok_count = labels_to_save.count('OK')
        ng_count = labels_to_save.count('NG')
        
        np.savez(save_path,
                 waveforms=segments_array,
                 labels=labels_array,
                 fs=st.session_state['fs'],
                 metric=st.session_state['metric_choice'],
                 auto_labels=np.array(st.session_state['labels']))  # 自動ラベルも保存
        
        st.success(f"✅ データセットを保存しました！")
        st.write(f"📁 保存先: {save_path}")
        st.write(f"📊 保存内容: OK: {ok_count}個, NG: {ng_count}個 (計{len(labels_to_save)}セグメント)")
        st.write(f"📍 作業ディレクトリ: {os.getcwd()}")
        
        # 手動編集の変更点があれば表示
        if 'manual_labels' in st.session_state:
            changed_count = sum(1 for i in range(len(st.session_state['labels'])) 
                              if st.session_state['manual_labels'][i] != st.session_state['labels'][i])
            if changed_count > 0:
                st.write(f"🔄 手動編集: {changed_count}個のラベルが変更されました")

    else:
        st.error("❌ 保存するデータが見つかりません。まずは録音を実施してください。")
