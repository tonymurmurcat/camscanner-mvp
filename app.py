import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="雲端掃描器 MVP v2", page_icon="📸")

# --- 核心圖像處理 (修改關鍵參數) ---
def process_document(image_array):
    # 1. 轉為灰階
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    # 2. 自適應二值化 (關鍵修改點!)
    # blockSize: 從 11 改為 31。更大的區域能更好地處理光照不均，避免把文字切碎。
    # C: 從 2 改為 15。這個值越大，背景會越白，有助於去除非文字的雜訊。
    processed_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    
    # 3. 輕微降噪 (選用，如果覺得字體邊緣太毛躁可以保留)
    processed_img = cv2.medianBlur(processed_img, 3)
    
    return processed_img

# --- UI 介面 ---
st.title("📸 雲端掃描器 MVP v2")
st.info("""
**拍攝小撇步以獲得最佳效果：**
1. ☀️ **光線充足**：在明亮的地方拍攝，避免陰影投射在文件上。
2. 📐 **保持平整**：盡量讓文件平鋪拍攝。
3. 📱 **拿穩手機**：點擊拍攝時保持穩定，避免照片模糊。
""")

# 呼叫手機相機
camera_image = st.camera_input("點擊拍攝文件")

if camera_image is not None:
    # 讀取圖片
    bytes_data = camera_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 顯示處理中狀態
    with st.spinner('正在進行優化處理...'):
        # 執行算法
        scanned_result = process_document(cv2_img)
        
        # 顯示結果對比
        st.subheader("處理結果")
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2_img, caption="原圖", channels="BGR", use_container_width=True)
        with col2:
            st.image(scanned_result, caption="掃描效果 (參數優化後)", use_container_width=True)

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
        file_name="my_scan_v2.pdf",
        mime="application/pdf",
        use_container_width=True
    )
