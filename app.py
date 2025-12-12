import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from streamlit_cropper import st_cropper  # 新增這個套件

st.set_page_config(page_title="全能掃描 MVP v6 (手動修正版)", page_icon="📐", layout="wide")

# --- 1. 核心邏輯區 ---

# 透視變換 (拉直)
def four_point_transform(image, pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# 自動偵測邊緣
def auto_detect_edge(image):
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image_small = cv2.resize(image, (int(image.shape[1] / ratio), 500))
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return four_point_transform(orig, approx.reshape(4, 2) * ratio), True
    return image, False

# 影像增強濾鏡
def enhance_image(image_array, mode, sharpen_val, contrast_val):
    # 確保輸入是灰階
    if len(image_array.shape) == 3:
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_array
        
    if mode == '模式 A: 智能增強 (推薦)':
        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        final_img = cv2.addWeighted(enhanced, 1.0 + sharpen_val, gaussian, -sharpen_val, 0)
        return final_img
    else:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 15)

# --- 2. 介面區 ---
st.title("📐 全能掃描 MVP v6 (手動修正版)")

with st.sidebar:
    st.header("🎛️ 設定控制")
    # 裁切模式選擇
    crop_mode = st.radio("裁切方式：", ('✨ 自動偵測 (Auto)', '🖐️ 手動框選 (Manual)'))
    
    st.markdown("---")
    st.write("**濾鏡調整：**")
    filter_mode = st.selectbox("濾鏡模式", ('模式 A: 智能增強 (推薦)', '模式 B: 純黑白'))
    sharpen = st.slider("銳化程度", 0.0, 3.0, 1.0)
    contrast = st.slider("對比度", 1.0, 5.0, 2.0)

uploaded_file = st.file_uploader("📤 請先拍照或上傳圖片", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # 讀取檔案為 PIL 格式 (供 Cropper 使用)
    pil_img = Image.open(uploaded_file)
    # 轉為 OpenCV 格式 (供自動演算法使用)
    cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    st.subheader("1️⃣ 裁切階段")
    
    cropped_result = None

    if crop_mode == '✨ 自動偵測 (Auto)':
        # 執行自動裁切
        processed_crop, success = auto_detect_edge(cv2_img)
        if success:
            st.success("成功偵測到邊緣！")
            st.image(processed_crop, caption="自動裁切結果", channels="BGR", use_container_width=True)
            cropped_result = processed_crop
        else:
            st.warning("⚠️ 自動偵測失敗，顯示原圖。請切換到「手動框選」模式。")
            st.image(cv2_img, caption="原始圖片", channels="BGR", use_container_width=True)
            cropped_result = cv2_img

    else: # 手動模式
        st.info("請在下方圖片上拖曳框線，選擇文件範圍。")
        # 呼叫手動裁切器
        cropped_box = st_cropper(
            pil_img,
            realtime_update=True,
            box_color='green',
            aspect_ratio=None
        )
        # 取得裁切後的圖片並轉回 OpenCV 格式
        cropped_result = cv2.cvtColor(np.array(cropped_box), cv2.COLOR_RGB2BGR)
        
        st.caption("預覽裁切後的效果：")
        st.image(cropped_result, channels="BGR", width=200)


    # --- 最終處理階段 ---
    if cropped_result is not None:
        st.markdown("---")
        st.subheader("2️⃣ 最終掃描結果")
        
        with st.spinner('正在進行影像增強...'):
            final_output = enhance_image(cropped_result, filter_mode, sharpen, contrast)
            st.image(final_output, caption="最終完成圖", use_container_width=True)

            # 下載
            result_pil = Image.fromarray(final_output)
            pdf_bytes = io.BytesIO()
            result_pil.save(pdf_bytes, format='PDF', resolution=150.0)
            
            st.download_button(
                label="📥 下載 PDF",
                data=pdf_bytes.getvalue(),
                file_name="scanned_v6.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )
