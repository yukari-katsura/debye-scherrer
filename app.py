import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.spatial.distance import pdist, squareform
import io
import os

# タイトル
st.title("Debye-Scherrer XRD Plots")
st.write("Developed by Yukari Katsura in 2024")

# データファイルのアップロード
data_files = st.file_uploader(
    "Upload powder-XRD data files",
    accept_multiple_files=True
)

# アップロードされたファイル数を表示
if data_files:
    st.write(f"Number of uploaded files: {len(data_files)}")

# ラベルファイルのアップロード (optional)
label_file = st.file_uploader(
    "Choose a file for labels (optional, requires 'filename' column; 'label' and 'order' columns optional)",
    type=["csv", "xlsx"]
)

# シート名の入力（Excelファイルの場合）
sheet_name = None
if label_file and label_file.name.endswith(".xlsx"):
    sheet_name = st.text_input("Enter sheet name (leave blank for default sheet)", "")

# カラーマップの選択肢を増やす
color_options = [
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu', 
    'PuBuGn', 'PuRd', 'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd',
    'afmhot', 'autumn', 'bone', 'cool', 'copper', 'gist_heat', 'gray', 'hot', 'pink',
    'spring', 'summer', 'winter', 'twilight', 'twilight_shifted', 'hsv', 'ocean',
    'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix',
    'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
]

# パラメータの指定部分をコンパクトに整理
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    width_cm = st.text_input("Plot width (cm)", "25")
with col2:
    height_cm = st.text_input("Plot height (cm)", "10")
with col3:
    dpi_options = [72, 150, 300, 600, 1200]
    dpi = st.selectbox("Resolution (DPI)", dpi_options)
with col4:
    cmap = st.selectbox("Colormap", color_options, index=color_options.index('Blues'))
with col5:
    format_options = ['png', 'jpeg', 'svg', 'pdf']
    img_format = st.selectbox("Image format", format_options)

col6, col7, col8, col9, col10 = st.columns(5)
with col6:
    col_x = st.text_input("Column for x", "")
with col7:
    col_y = st.text_input("Column for y", "")
with col8:
    x_step_options = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]
    x_step = st.selectbox("Step for x", x_step_options, index=4)
with col9:
    scale_options = ['linear', 'log']
    scale = st.selectbox("Color scale", scale_options)
with col10:
    sort_options = ['None', 'File', 'Similarity']
    sort_method = st.selectbox("Sort method", sort_options)

# ラベルファイルがアップロードされた場合
labels = {}
order = []
if label_file:
    if label_file.name.endswith(".csv"):
        labels_df = pd.read_csv(label_file)
    elif label_file.name.endswith(".xlsx"):
        if sheet_name:
            labels_df = pd.read_excel(label_file, sheet_name=sheet_name)
        else:
            labels_df = pd.read_excel(label_file)

    if 'order' in labels_df.columns:
        labels_df = labels_df.sort_values('order')  # order列でソート
    for index, row in labels_df.iterrows():
        label = row['filename']
        if 'label' in row:
            label = row['label']
        labels[row['filename']] = label
        order.append(row['filename'])

# プロットを行うボタン
if st.button("Plot Heatmap"):
    # ヒートマップ用のデータフレーム
    heatmap_data = pd.DataFrame()
    x_ticks = None  # x軸の目盛りラベル

    if data_files:
        data_dict = {data_file.name: data_file for data_file in data_files}
        if sort_method == 'File' and order:
            sorted_data_files = [data_dict[file] for file in order if file in data_dict]
        else:
            sorted_data_files = list(data_files)

        for data_file in sorted_data_files:
            # データの読み込み
            file_content = data_file.read().decode('utf-8')
            lines = file_content.splitlines()
            header_lines = [line for line in lines if line.startswith("#") or line.startswith("'")]
            data_lines = [line for line in lines if not (line.startswith("#") or line.startswith("'"))]
            
            # ヘッダー行の処理
            if len(header_lines) == 1:
                df = pd.read_csv(io.StringIO("\n".join([header_lines[0]] + data_lines)), delimiter=r'\s+', header=0)
            else:
                df = pd.read_csv(io.StringIO("\n".join(data_lines)), delimiter=r'\s+', header=None)

            # 列数が足りない場合のチェック
            if df.shape[1] < 2:
                st.warning(f"The file {data_file.name} does not have enough columns.")
                continue
            
            # 列名が指定されていない場合のデフォルト設定
            if col_x == "" or col_x not in df.columns:
                col_x = df.columns[0]
            if col_y == "" or col_y not in df.columns:
                col_y = df.columns[1] if len(df.columns) > 1 else None

            if col_y is None:
                st.warning(f"The file {data_file.name} does not have enough columns.")
                continue

            # スプライン補間の実行
            try:
                x = df[col_x].astype(float)
                y = df[col_y].astype(float)
            except ValueError:
                st.error(f"Failed to convert columns {col_x} and {col_y} to float in file {data_file.name}.")
                continue

            # スケールの適用
            if scale == 'log':
                y = np.log1p(y)  # 自然対数を取る（+1は値が0の時の対処）
                # 無限大やNaNを補完
                y.replace([np.inf, -np.inf], np.nan, inplace=True)
                y.fillna(y.max(), inplace=True)  # NaNを最大値で補完

            # 補間のための新しいx軸の生成
            x_start = np.ceil(x.min() / x_step) * x_step
            x_new = np.arange(x_start, x.max() + x_step, x_step)
            if len(x) != len(y):
                st.error(f"Lengths of x and y do not match after filtering in file {data_file.name}.")
                continue
            spline = make_interp_spline(x, y)
            y_new = spline(x_new)
            
            # x軸の目盛りラベルを設定
            if x_ticks is None:
                x_ticks = np.arange(np.floor(x.min() / 10) * 10, np.ceil(x.max() / 10) * 10 + 10, 10)
            
            # 補間結果をデータフレームに追加
            label = labels.get(data_file.name, os.path.splitext(data_file.name)[0])
            heatmap_data[label] = y_new

        if heatmap_data.empty:
            st.error("No valid data found in the uploaded files.")
        else:
            # 類似性に基づくソート
            if sort_method == 'Similarity':
                distance_matrix = squareform(pdist(heatmap_data.T, metric='euclidean'))
                sorted_indices = np.argsort(distance_matrix.sum(axis=1))
                sorted_heatmap_data = heatmap_data.iloc[:, sorted_indices]
            else:
                sorted_heatmap_data = heatmap_data

            # cmをinchに変換
            width_inch = float(width_cm) / 2.54
            height_inch = float(height_cm) / 2.54

            # ヒートマップの作成
            fig, ax = plt.subplots(figsize=(width_inch, height_inch))
            sns.heatmap(sorted_heatmap_data.T, cmap=cmap, ax=ax)

            # x軸の目盛りを10刻みの値に設定
            ax.set_xticks(np.linspace(0, len(x_new) - 1, len(x_ticks)))
            ax.set_xticklabels([f'{int(tick)}' for tick in x_ticks], rotation=0)

            # ヒートマップの表示
            st.pyplot(fig)

            # 画像のピクセル数を計算
            width_px = width_inch * dpi
            height_px = height_inch * dpi
            st.write(f"Image dimensions: {int(width_px)} x {int(height_px)} pixels")

            # プロットの画像をバッファに保存
            buf = io.BytesIO()
            fig.savefig(buf, format=img_format, dpi=dpi)
            buf.seek(0)

            # ダウンロードボタンを配置
            st.download_button(
                label=f"Download Heatmap as {img_format.upper()}",
                data=buf,
                file_name=f"heatmap.{img_format}",
                mime=f"image/{img_format}"
            )
