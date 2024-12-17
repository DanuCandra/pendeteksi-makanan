import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, decode_predictions, preprocess_input
from fuzzywuzzy import process  # Library untuk fuzzy matching
import pandas as pd
import sys

# Memuat model pre-trained ResNet50
model = ResNet50(weights='imagenet')

# Memuat dataset makanan yang telah dibersihkan
try:
    # Membaca dataset
    file_path = r"C:\\Users\\danuc\\OneDrive\\Documents\\SEMESTER5\\Sistem Cerdas\\FINAL\\FOOD\\final.csv"
    food_data_cleaned = pd.read_csv(file_path, sep=';')

    # Membersihkan dataset
    food_data_cleaned = food_data_cleaned.rename(columns=lambda x: x.strip() if isinstance(x, str) else x)
    columns_to_keep = ['food', 'Caloric Value', 'Sugars', 'Protein']
    food_data_cleaned = food_data_cleaned[columns_to_keep]
    food_data_cleaned = food_data_cleaned.dropna().reset_index(drop=True)
    food_data_cleaned['food'] = food_data_cleaned['food'].str.lower().str.strip()
    food_nutrition = food_data_cleaned.set_index('food').T.to_dict()
except Exception as e:
    print(f"Error loading dataset: {e}")
    food_nutrition = {}

# Membuka webcam
cap = cv2.VideoCapture(0)

# Mengecek jika webcam terbuka dengan benar
if not cap.isOpened():
    print("Tidak dapat mengakses webcam.")
    sys.exit()

# Membuka jendela untuk deteksi makanan
cv2.namedWindow("Deteksi Makanan", cv2.WINDOW_NORMAL)
fullscreen = False

while True:
    # Membaca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil gambar dari webcam.")
        break

    # Membuat salinan frame untuk menampilkan teks
    display_frame = frame.copy()

    try:
        # Memproses gambar untuk model ResNet50
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengonversi ke RGB
        img = cv2.resize(img, (224, 224))  # Mengubah ukuran gambar menjadi 224x224 (input ResNet50)
        img = np.expand_dims(img, axis=0)  # Menambah dimensi batch
        img = preprocess_input(img)  # Preprocessing sesuai model ResNet50

        # Prediksi dengan model ResNet50
        preds = model.predict(img)
        decoded_preds = decode_predictions(preds, top=1)[0]

        # Mendapatkan nama makanan yang diprediksi dan mengubahnya menjadi huruf kecil
        predicted_food_name = decoded_preds[0][1].lower().strip()

        # Mencari nama yang paling mirip dengan nama yang diprediksi menggunakan fuzzywuzzy
        if food_nutrition:
            best_match = process.extractOne(predicted_food_name, list(food_nutrition.keys()))
            matched_food = best_match[0]

            # Mendapatkan informasi gizi dari dataset makanan
            nutrition = food_nutrition.get(matched_food, {})
            calories = nutrition.get('Caloric Value', 'Tidak Diketahui')
            sugar = f"{nutrition.get('Sugars', 'Tidak Diketahui')} g"
            protein = f"{nutrition.get('Protein', 'Tidak Diketahui')} g"
        else:
            matched_food = "Tidak Diketahui"
            calories = sugar = protein = "Dataset Kosong"

        # Menampilkan informasi makanan pada frame
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
        display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)

        cv2.putText(display_frame, f'Makanan: {matched_food}', 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Kalori: {calories}', 
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Gula: {sugar}', 
                    (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f'Protein: {protein}', 
                    (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error processing frame: {e}")
        cv2.putText(display_frame, "Error processing image", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Menampilkan frame
    cv2.imshow('Deteksi Makanan', display_frame)

    # Buat Keluar Aplikasi
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # Tombol ESC untuk keluar
        break
    elif key == ord('f'):  # Tombol 'f' untuk toggle fullscreen
        fullscreen = not fullscreen
        if fullscreen:
            cv2.setWindowProperty("Deteksi Makanan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("Deteksi Makanan", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

# Menutup webcam
cap.release()
cv2.destroyAllWindows()
sys.exit()
