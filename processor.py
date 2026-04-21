import cv2
import imutils
import numpy as np
from scipy.spatial import distance as dist

def measure_egg(image_path, reference_width_cm=2.5):
    # 1. โหลดภาพและเตรียมภาพ
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2. หาขอบวัตถุ
    edged = cv2.Canny(blur, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # 3. หา Contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if not cnts:
        return None, "ไม่พบวัตถุในภาพ"

    # เรียงลำดับวัตถุจากซ้ายไปขวา (สมมติว่าวัตถุซ้ายสุดคือเหรียญอ้างอิง)
    (cnts, _) = imutils.contours.sort_contours(cnts)
    pixelsPerMetric = None
    output_img = image.copy()

    for c in cnts:
        # กรองวัตถุขนาดเล็กเกินไปทิ้ง (Noise)
        if cv2.contourArea(c) < 500:
            continue

        # วาดกล่องรอบวัตถุ
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        cv2.drawContours(output_img, [box], -1, (0, 255, 0), 2)

        # คำนวณความกว้างยาวเป็น Pixel
        (x, y), (w, h), angle = rect
        
        # ถ้ายังไม่มีค่าอ้างอิง ให้ใช้วัตถุแรกเป็นตัวตั้งต้น (เช่น เหรียญ 10 บาทกว้าง 2.6 ซม.)
        if pixelsPerMetric is None:
            pixelsPerMetric = max(w, h) / reference_width_cm
            continue

        # คำนวณขนาดจริง
        width_cm = w / pixelsPerMetric
        height_cm = h / pixelsPerMetric

        # เขียนตัวเลขลงในภาพ
        cv2.putText(output_img, f"{width_cm:.1f}cm x {height_cm:.1f}cm", 
                    (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return output_img, "ประมวลผลเสร็จสิ้น"