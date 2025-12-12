import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import io

st.set_page_config(page_title="雲端掃描器 MVP v3", page_icon="🖨️", layout="wide")

# --- 側邊欄控制參數 ---
st.sidebar.header("🎛️ 影像調整中心")
st.sidebar.write("如果結果不理想，試著調整這裡。")

# 模式選擇
scan_mode = st.sidebar.radio(
    "處理模式選擇：",
    ('模式 A: 高對比灰階 (推薦)', '模式 B: 純黑白二值化 (舊版)')
)

st.sidebar.markdown("---")

# 模式 A 的參數
if scan_mode == '模式 A: 高對比灰階 (推薦)':
    st.sidebar.subheader("模式 A 參數微調")
    # CLAHE Clip Limit: 控制對比度增強的程度。越高對比越強，但雜訊也越多。
    clahe_clip = st.sidebar.slider("對比度增強 (Clip Limit)", 1.0, 10.0, 3.0, 0.5)
    # 銳化程度
    sharpen_amount = st.sidebar.slider("銳化程度", 0.0, 5.0, 1.5, 0.1)

# 模式 B 的參數
else:
    st.sidebar.subheader("模式 B 參數微調")
    st.sidebar.info("此模式需要光線非常充足且對焦清晰的照片。")
    # Block Size: 決定局部閾值的區域大小。必須是奇數。
    block_size = st.sidebar.slider("區域大小 (Block Size)", 11, 101, 51, 2)
    # C: 常數，從平均值中減去的值。越大背景越白。
    c_value = st.sidebar.slider("背景常數 (C)", 1, 50, 15, 1)


# --- 核心圖像處理算法 ---
def process_image_v3(image_array, mode, clip_limit, sharpen, blk_size, c_val):
    # 1. 轉換為灰階
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    if mode == '模式 A: 高對比灰階 (推薦)':
        # --- 新算法: CLAHE + 銳化 ---
        
        # 步驟 2: 應用 CLAHE (增強局部對比度，拯救陰影)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 步驟 3: 影像銳化 (Unsharp Masking 原理)
        # 先做一個高斯模糊版本
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        # 公式: 原始 * (1+銳化度) - 模糊 * 銳化度
        sharpened = cv2.addWeighted(enhanced, 1.0 + sharpen, gaussian, -sharpen, 0)
        
        return sharpened

    else:
        # --- 舊算法: 自適應二值化 (給光線極好時用) ---
        blurred = cv2.medianBlur(gray, 3)
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk_size, c_val
        )
        return binary


# --- 主介面 ---
st.title("🖨️ 雲端掃描器 MVP v3 (高對比版)")
st.markdown("""
如果是模糊或光線昏暗的照片，請使用預設的 **「模式 A」**，並嘗試調整側邊欄的滑桿。
""")

# 呼叫手機相機
camera_image = st.camera_input("📸 點擊拍攝文件")

if camera_image is not None:
    # 讀取圖片
    bytes_data = camera_image.getvalue()
    original_cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    # 執行處理
    with st.spinner('正在執行影像增強演算法...'):
        processed_result = process_image_v3(
            original_cv2_img,
            scan_mode,
            clahe_clip if scan_mode == '模式 A: 高對比灰階 (推薦)' else 0,
            sharpen_amount if scan_mode == '模式 A: 高對比灰階 (推薦)' else 0,
            block_size if scan_mode != '模式 A: 高對比灰階 (推薦)' else 0,
            c_value if scan_mode != '模式 A: 高對比灰階 (推薦)' else 0
        )

    # 顯示結果比較 (使用較寬的佈局)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("原始照片")
        st.image(original_cv2_img, channels="BGR", use_container_width=True)
    with col2:
        st.subheader(f"處理結果 ({scan_mode[:4]})")
        st.image(processed_result, caption="可透過側邊欄微調效果", use_container_width=True)

    # 產生 PDF 下載
    pil_img = Image.fromarray(processed_result)
    pdf_bytes = io.BytesIO()
    pil_img.save(pdf_bytes, format='PDF', resolution=150.0)
    pdf_data = pdf_bytes.getvalue()

    st.download_button(
        label="📥 下載處理後的 PDF",
        data=pdf_data,
        file_name="enhanced_scan.pdf",
        mime="application/pdf",
        use_container_width=True,
        type="primary"
    )
