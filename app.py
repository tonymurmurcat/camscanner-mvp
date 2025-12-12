import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

st.set_page_config(page_title="全能掃描 MVP v5 (自動裁切版)", page_icon="✂️", layout="wide")

# --- 核心幾何演算法 (處理透視變換) ---
def order_points(pts):
    # 重新排列四個點的順序：左上, 右上, 右下, 左下
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # 左上
    rect[2] = pts[np.argmax(s)] # 右下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # 右上
    rect[3] = pts[np.argmax(diff)] # 左下
    return rect

def four_point_transform(image, pts):
    # 取得鳥瞰圖 (Top-down view)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 計算新圖片的寬度與高度
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 建構目標點
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 計算透視變換矩陣並應用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_scan_document(image):
    # 1. 邊緣檢測前處理
    # 縮小圖片以加速偵測 (處理完後會映射回原圖)
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image_small = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 邊緣檢測
    edged = cv2.Canny(blurred, 75, 200)

    # 2. 尋找輪廓
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    # 遍歷輪廓，找最大的四邊形
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    # 3. 如果找到四邊形，進行透視裁切
    if screenCnt is not None:
        warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
        return warped, True
    else:
        # 沒找到明顯邊界，返回原圖
        return image, False

# --- 影像增強演算法 (沿用上一版的成功經驗) ---
def enhance_image(image_array, mode, sharpen_val, contrast_val):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    
    if mode == '模式 A: 智能增強 (推薦)':
        clahe = cv2.createCLAHE(clipLimit=contrast_val, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3.0)
        final_img = cv2.addWeighted(enhanced, 1.0 + sharpen_val, gaussian, -sharpen_val, 0)
        return final_img
    else: # 純黑白
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 15)

# --- Streamlit UI ---
st.title("✂️ 全能掃描 MVP v5 (自動裁切版)")

# 側邊欄
with st.sidebar:
    st.header("設定")
    enable_crop = st.checkbox("啟用自動裁切 (Auto-Crop)", value=True)
    st.info("如果自動裁切切壞了，請取消上面的勾選。")
    
    scan_mode = st.radio("濾鏡模式：", ('模式 A: 智能增強 (推薦)', '模式 B: 純黑白'))
    sharpen = st.slider("銳化程度", 0.0, 3.0, 1.0)
    contrast = st.slider("對比度", 1.0, 5.0, 2.0)

uploaded_file = st.file_uploader("📤 拍照或上傳圖片", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_img, caption="原始照片", channels="BGR", use_container_width=True)

    with st.spinner('AI 正在分析文件邊緣並進行處理...'):
        # 步驟 1: 自動裁切
        if enable_crop:
            cropped_img, success = auto_scan_document(original_img)
            if success:
                st.toast("✅ 成功偵測到文件邊緣！", icon="✂️")
            else:
                st.toast("⚠️ 找不到明顯邊緣，使用原圖。", icon="🔍")
        else:
            cropped_img = original_img

        # 步驟 2: 影像增強
        final_result = enhance_image(cropped_img, scan_mode, sharpen, contrast)

    with col2:
        st.image(final_result, caption="最終掃描結果", use_container_width=True)

    # 下載
    pil_img = Image.fromarray(final_result)
    pdf_bytes = io.BytesIO()
    pil_img.save(pdf_bytes, format='PDF', resolution=150.0)
    
    st.download_button(
        label="📥 下載 PDF",
        data=pdf_bytes.getvalue(),
        file_name="scanned_doc.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True
    )
