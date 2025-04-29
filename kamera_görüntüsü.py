import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque


# === MODEL YÜKLE ===
model = load_model("hayvan_siniflandirma_modeli2.h5")

# === SINIF İSİMLERİ ===
class_names = [
    "cat","goat","hen", "sheep","elephant",
    "dog", "rabbit", "cow", "chicken","deer","horse"
]
#model iki köpek,fil,geyik
#model bir tavşan
#model tavuk inek
# === TÜRKÇE KARŞILIKLAR ===
translate_tr = {
    "horse":"at",
    "elephant":"fil",
    "cat": "kedi",
    "sheep": "koyun",
    "dog": "kopek",
    "hen": "tavuk",
    "rabbit": "tavsan",
    "cow": "inek",
    "goat":"keci",
    "chicken": "tavuk",
    "deer": "geyik"
}

# === GÖRÜNTÜ BOYUTU VE DİĞER AYARLAR ===
img_size = 128
color = (0, 0, 255)  # Kırmızı
prediction_buffer = deque(maxlen=5)  # Son 5 tahmini tutar

# === KAMERA BAĞLANTISI ===
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("❌ Kamera açılamadı!")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("❌ Kamera görüntüsü alınamadı.")
        break

    # === GÖRÜNTÜ ÖN İŞLEME ===
    frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)  # Gürültüyü azalt
    frame_rgb = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)  # RGB formatına dönüştür
    frame_resized = cv2.resize(frame_rgb, (img_size, img_size))  # 128x128 boyutuna indir
    normalized = frame_resized / 255.0  # Normalize et
    input_data = np.expand_dims(normalized.astype(np.float32), axis=0)  # Batch boyutu ekle

    # === TAHMİN ===
    prediction = model.predict(input_data, verbose=0)[0]
    prediction_buffer.append(prediction)  # Son tahminleri biriktir

    # === ORTALAMA TAHMİN (Kararlılık için) ===
    avg_prediction = np.mean(prediction_buffer, axis=0)
    predicted_index = np.argmax(avg_prediction)
    confidence = avg_prediction[predicted_index]
    class_en = class_names[predicted_index]
    class_tr = translate_tr.get(class_en, "bilinmiyor")

    # === METNİ EKRANA YAZ ===
    text = f"{class_tr.upper()} ({confidence * 100:.1f}%)"
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, color, 3)

    # === GÖRÜNTÜYÜ GÖSTER ===
    cv2.imshow("Canli Hayvan Tanima", frame)

    # ESC tuşuna basılırsa çık
    if cv2.waitKey(1) == 27:
        break

# === TEMİZLİK ===
cam.release()
cv2.destroyAllWindows()
