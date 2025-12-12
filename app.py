import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# 設定網頁標題
st.set_page_config(page_title="高清掃描器 MVP v4", page_icon="📄", layout="wide")

# --- 介面引導 ---
st.title("📄 高清掃描器 MVP v4")
st.markdown("""
**💡 改進說明：** 為了看清楚 A4 文件的小字，我們改用**「原生相機」**模式。
請點擊下方的 **「Browse files」**，然後選擇 **「拍照 (Take Photo)」**。
這樣可以使用手機的自動對焦和最高畫質。
""")

# --- 1. 側邊欄在哪裡？ (解決你的問題 3) ---
with st.sidebar:
    st.header("🎛️ 影像調整")
    st.info("👈 手機版請點擊左上角的「>」箭頭來展開這個選單。")
    
    # 模式選擇
    scan_mode = st.radio(
        "處理模式：",
        ('模式 A: 智能增強 (推薦)', '模式 B: 高對比二值化')
    )
    
    st.markdown("---")
    if scan_mode == '模式 A: 智能增強 (推薦)':
        st.write("**增強參數微調：**")
        sharpen = st.slider("銳化程度", 0.0, 3.0, 1.0, 0.1)
        contrast = st.slider("對比度", 1.0, 5.0, 2.0, 0.2)
    else:
        st.write("**黑白參數微調：**")
        # 因為原生相機畫素很高，Block Size 需要設很大
        block_size = st.slider("區域大小 (Block Size)", 21, 201, 91, 2)
        c_val = st.slider("去噪強度 (C)", 1, 50, 15, 1)

# --- 2. 核心算法 (針對高畫質優化) ---
def process_image_high_res(image_array, mode, sharpen_val, contrast_val, blk, c):
    # 轉灰階
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    if mode == '模式 A: 智能增強 (推薦)':
        # CLAHE (限制對比度自適應直方圖均衡化)
        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 銳化濾鏡 (Unsharp Mask)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        final_img = cv2.addWeighted(enhanced, 1.0 + sharpen_val, gaussian, -sharpen_val, 0)
        return final_img
        
    else:
        # 自適應二值化 (Adaptive Threshold)
        # 針對高畫素圖片，先做一點高斯模糊降噪
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, c
        )
        return binary

# --- 3. 檔案上傳區 (取代原本的 camera_input) ---
uploaded_file = st.file_uploader("📤 點此啟動相機或上傳圖片", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # 讀取圖片
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_cv2_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 顯示原始資訊 (確認解析度)
    h, w, _ = original_cv2_img.shape
    st.caption(f"原始解析度：{w} x {h} (畫素越高，小字越清楚)")

    # 處理圖片
    with st.spinner('正在使用高畫質演算法處理...'):
        
        # 根據側邊欄參數處理
        if scan_mode == '模式 A: 智能增強 (推薦)':
            processed_result = process_image_high_res(original_cv2_img, scan_mode, sharpen, contrast, 0, 0)
        else:
            processed_result = process_image_high_res(original_cv2_img, scan_mode, 0, 0, block_size, c_val)

        # 顯示結果
        st.subheader("處理結果")
        st.image(processed_result, caption="高清掃描結果", use_container_width=True)

        # 轉 PDF
        pil_img = Image.fromarray(processed_result)
        pdf_bytes = io.BytesIO()
        pil_img.save(pdf_bytes, format='PDF', resolution=150.0)
        
        st.download_button(
            label="📥 下載高清 PDF",
            data=pdf_bytes.getvalue(),
            file_name="high_res_scan.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
