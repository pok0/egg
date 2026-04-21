import cv2

# โหลดรูปภาพ (อย่าลืมเอาไฟล์รูปไข่ใส่ไว้ในโฟลเดอร์ testeggs ด้วยนะครับ)
image_path = "eggs.jpg" # เปลี่ยนชื่อให้ตรงกับไฟล์ของคุณ
image = cv2.imread(image_path)

if image is None:
    print("ไม่พบไฟล์รูปภาพ กรุณาตรวจสอบชื่อไฟล์")
else:
    # ลองแสดงผลรูปภาพ
    cv2.imshow("Test Egg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()