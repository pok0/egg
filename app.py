import streamlit as st
import cv2
import numpy as np
from PIL import Image
import imutils
from imutils import contours  # แก้ไข: เพิ่มการ import contours แยกออกมา
from scipy.spatial import distance as dist

# ฟังก์ชันสำหรับประมวลผลภาพ
def process_image(image, ref_width):
    # 1. เตรียมภาพ
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. ตรวจจับขอบ (ปรับค่า Threshold ให้เหมาะสมกับไข่)
    edged = cv2.Canny(gray, 50, 100)
    # edged = cv2.Canny(gray, 100, 200)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 3. หา Contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # กรองเอาเฉพาะวัตถุที่มีขนาดใหญ่พอ (ป้องกัน Noise)
    valid_cnts = [c for c in cnts if cv2.contourArea(c) > 500]

    if len(valid_cnts) < 2:
        return None, "ต้องมีวัตถุอย่างน้อย 2 ชิ้น (วัตถุอ้างอิงอยู่ซ้ายสุด และตามด้วยไข่)"

    # 4. เรียงจากซ้ายไปขวา (วัตถุอ้างอิงต้องอยู่ซ้ายสุดเสมอ)
    (cnts, _) = contours.sort_contours(valid_cnts)
    pixelsPerMetric = None
    
    for c in cnts:
        # คำนวณหาล้อมกรอบสี่เหลี่ยมแบบหมุนตามวัตถุ
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        
        # วาดเส้นรอบวัตถุ
        cv2.drawContours(img, [box], -1, (0, 255, 0), 2)
        
        # ดึงค่าตำแหน่งและความกว้างยาว (พิกเซล)
        (x, y), (w, h), angle = rect
        
        # กรณีวัตถุแรก (ซ้ายสุด) ใช้เป็น Reference
        if pixelsPerMetric is None:
            # ใช้ค่าที่มากที่สุดของด้าน (เหรียญควรจะกลมหรือเป็นสี่เหลี่ยมจัตุรัส)
            pixelsPerMetric = max(w, h) / ref_width
            continue
            
        # 5. คำนวณขนาดจริงสำหรับไข่
        # โดยปกติไข่จะวัดความกว้าง (จุดที่อ้วนที่สุด) และความยาว
        dimA = max(w, h) / pixelsPerMetric # ความยาว
        dimB = min(w, h) / pixelsPerMetric # ความกว้าง
        
        # เขียนตัวเลขลงบนภาพ
        cv2.putText(img, f"L: {dimA:.2f}cm", (int(x), int(y)-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(img, f"W: {dimB:.2f}cm", (int(x), int(y)+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "วิเคราะห์สำเร็จ"

# --- ส่วนของ UI (Streamlit) ---
st.set_page_config(page_title="Egg Size AI", page_icon="🥚")
st.title("🥚 Egg Size Measurement AI")
st.markdown("---")

# Sidebar สำหรับตั้งค่า
st.sidebar.header("Settings")
ref_size = st.sidebar.number_input("ขนาดวัตถุอ้างอิง (ซม.)", value=2.6, help="เช่น เหรียญ 10 บาทไทย = 2.6 ซม.")

uploaded_file = st.file_uploader("อัปโหลดรูปภาพไข่ (วางเหรียญไว้ด้านซ้ายสุดของภาพ)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    
    # แสดงรูปต้นฉบับในคอลัมน์ซ้าย และผลลัพธ์ในคอลัมน์ขวา
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(input_image, caption="รูปต้นฉบับ", use_container_width=True)
    
    if st.button("เริ่มการวัดขนาด 🚀"):
        result_img, msg = process_image(input_image, ref_size)
        
        with col2:
            if result_img is not None:
                st.image(result_img, caption="ผลการประมวลผล", use_container_width=True)
                st.success(msg)
            else:
                st.error(msg)