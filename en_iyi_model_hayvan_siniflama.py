import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from translate import translate  # Türkçe → İngilizce sınıf çevirileri

# === 1. Veri Yolu ve Kontrol ===
data_dir = r"Dataset_Of_Animal_Images"
img_size = 128

if not os.path.exists(data_dir):
    raise FileNotFoundError(f"❌ Belirtilen klasör yolu bulunamadı: {data_dir}")

images, labels = [], []
label_map = {}
class_names = []

# === 2. Görüntüleri Yükle ===
for folder in os.listdir(data_dir):
    image_folder = os.path.join(data_dir, folder, "train", "images")
    if not os.path.isdir(image_folder):
        continue

    label_en = translate.get(folder.lower())
    if label_en is None:
        print(f"❌ translate içinde bulunamadı: {folder}")
        continue

    for file in os.listdir(image_folder):
        if file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_folder, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0

            if label_en not in label_map:
                label_map[label_en] = len(label_map)
                class_names.append(label_en)

            images.append(img)
            labels.append(label_map[label_en])

# === 3. Veriyi Numpy'a Çevir ===
X = np.array(images)
y = to_categorical(np.array(labels), num_classes=len(class_names))

# === 4. Eğitim/Test Bölmesi ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Veri Arttırma ===
train_aug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)
val_aug = ImageDataGenerator()

train_generator = train_aug.flow(X_train, y_train, batch_size=32)
val_generator = val_aug.flow(X_test, y_test, batch_size=32)

# === 6. CNN Modeli Kur ===
model = Sequential([
    Input(shape=(img_size, img_size, 3)),

    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 7. Modeli Eğit ===
history = model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)]
)

# === 8. Performans Grafiği ===
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.title('Model Eğitimi Başarımı')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === 9. MODELİ DOĞRU ŞEKİLDE KAYDET ===
#model.save("hayvan_siniflandirma_modeli")  # SavedModel formatında klasör olarak kaydediliyor

# Eğer illa .h5 formatı istersek:
model.save("hayvan_siniflandirma_modeli1.h5", include_optimizer=True, save_format="h5")

# === 10. Sınıf Etiketlerini Yazdır ===
print("\n✅ Eğitim tamamlandı. Sınıf Etiketleri:")
for label, idx in label_map.items():
    print(f"{idx}: {label}")
