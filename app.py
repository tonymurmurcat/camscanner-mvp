import streamlit as st
import cv2
import numpy as np
from PIL import Image
import img2pdf
import io

st.set_page_config(page_title="雲端掃描器 MVP", page_icon="📸")

# --- 核心圖像處理 ---
def process_document(image_array):
    # 1. 轉灰階
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    # 2. 自適應二值化 (模擬掃描效果)
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    # 3. 降噪
    processed_img = cv2.medianBlur(processed_img, 3)
    return processed_img

# --- UI 介面 ---
st.title("📸 雲端掃描器 MVP")
st.info("請使用手機豎屏拍攝，盡量保持文件平整。")

# 呼叫手機相機
camera_image = st.camera_input("點擊拍攝文件")

if camera_image is not None:
    # 讀取圖片
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 顯示處理中狀態
    with st.spinner('正在進行圖像處理...'):
        # 執行算法
        scanned_result = process_document(cv2_img)
        
        # 顯示結果對比
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_img, caption="原圖", channels="BGR")
        with col2:
            st.image(scanned_result, caption="掃描效果")

        # 轉換為 PDF
        pil_img = Image.fromarray(scanned_result)
        pdf_bytes = io.BytesIO()
        pil_img.save(pdf_bytes, format='PDF', resolution=100.0)
        pdf_data = pdf_bytes.getvalue()

    st.success("處理完成！")
    
    # 下載按鈕
    st.download_button(
        label="📥 下載 PDF 檔案",
        data=pdf_data,
        file_name="my_scan.pdf",
        mime="application/pdf",
        use_container_width=True 
    )