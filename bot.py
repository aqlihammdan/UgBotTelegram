import pandas as pd
import numpy as np
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Logging untuk debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load dataset dan training model
data = [
    ("Halo", "sapaan"),
    ("Halo tolong bantu", "sapaan"),
    ("Halo permisi", "sapaan"),
    ("Halo bantu saya", "sapaan"),
    ("Halo chatbot", "sapaan"),
    ("Dimana alamat kampus gundar?", "alamat"),
    ("Gundar dimana?", "alamat"),
    ("Ug ada dimana", "alamat"),
    ("Alamat kampus?", "alamat"),
    ("Alamat", "alamat"),
    ("Kampus gundar dimana?", "alamat"),
    ("Jurusan", "jurusan"),
    ("Jurusan yang ada di gundar", "jurusan"),
    ("Jurusan di gundar ada apa aja?", "jurusan"),
    ("Jurusannya apa aja", "jurusan"),
    ("Ada apa aja jurusannya?", "jurusan"),
    ("Bayaran", "bayaran"),
    ("Bayaran gundar berapa?", "bayaran"),
    ("Bayarannya berapa?", "bayaran"),
    ("Uktnya berapa", "bayaran"),
    ("Per semester berapa bayarannya", "bayaran"),
    ("Untuk ukt di kampus gunadarma berapa ?", "bayaran"),
    ("Berapa biaya masuk gunadarma?", "bayaran"),
    ("Jadwal", "jadwal"),
    ("Jadwal kuliah", "jadwal"),
    ("Jadwal kuliah di gundar", "jadwal"),
    ("Jadwal perkuliahan", "jadwal"),
    ("Bagaimana jika saya ingin melihat jadwal", "jadwal"),
    ("Melihat jadwal", "jadwal"),
    ("Kalender", "kalender"),
    ("Kalender akademik", "kalender"),
    ("Mau liat kalender akademik", "kalender"),
    ("Bagaimana kalender akademik gundar?", "kalender"),
    ("Mau lihat kalender akademik.", "kalender"),
    ("Cuti", "cuti"),
    ("Cara cuti", "cuti"),
    ("Pengajuan cuti", "cuti"),
    ("Bagaimana cara mengajukan cuti?", "cuti"),
    ("Caranya cuti pada kampus gundar?", "cuti"),
    ("Cuti akademik", "cuti"),
    ("Terima kasih", "makasih"),
    ("Thank you", "makasih"),
    ("Terima kasih atas informasinya.", "makasih"),
    ("Makasih sudah membantu", "makasih"),
    ("Baik terima kasih.", "makasih"),
    ("Sudah", "penutup"),
    ("Sudah jelas", "penutup"),
    ("Jelas", "penutup"),
    ("Informasi yang diberikan sudah jelas.", "penutup"),
    ("Infonya udah jelas.", "penutup")
]

df = pd.DataFrame(data, columns=["text", "label"])

# Vectorisasi teks
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Fungsi untuk melakukan preprocessing pada input
def preprocess_input(text):
    # Mengubah teks menjadi huruf kecil
    text = text.lower()
    # Menghapus karakter yang bukan huruf atau angka
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Menghapus spasi tambahan
    text = text.strip()
    return text

# Fungsi untuk mendeteksi apakah input tidak jelas
def is_gibberish(text):
    # Deteksi input yang tidak jelas berdasarkan panjang kata dan huruf berulang
    if len(text) > 15 and re.match(r'^[a-zA-Z]+$', text):
        if len(set(text)) < 4:  # Misal, hanya ada kurang dari 4 karakter unik
            return True
    return False

# Fungsi untuk prediksi kategori
def predict_category(text):
    text = preprocess_input(text)
    if is_gibberish(text):
        return "unknown"
    logger.info(f"User Input: {text}")
    X_test = vectorizer.transform([text])
    probs = model.predict_proba(X_test)[0]
    max_prob = np.max(probs)
    predicted_category = model.classes_[np.argmax(probs)]
    
    # Tentukan threshold untuk menangani input yang tidak jelas
    threshold = 0.24
    if max_prob >= threshold:
        logger.info(f"Predicted Category: {predicted_category} with confidence: {max_prob}")
        return predicted_category
    else:
        logger.info(f"Prediction below threshold with confidence: {max_prob}. Returning unknown.")
        return "unknown"

# Fungsi untuk mendapatkan respons sesuai kategori
def get_response(category):
    responses = {
        "sapaan": "Halo, Apakah ada yang bisa UgBot bantu? atau gunakan /help untuk melihat informasi apa saja yang dapat diberikan oleh Bot.",
        "alamat": "Untuk lokasi Universitas Gunadarma terletak di berbagai wilayah : \n 1. Kampus A (Jl. Kenari nomor 13 Jakarta Pusat, 10430 Phone : 330220, 330226) \n 2. Kampus B (Jl. Salemba Bluntas Jakarta Pusat) \n 3. Kampus C (Jl. Salemba Raya nomor 53 Jakarta Pusat Phone : 3906518, 3908568 Fax : 3100325) \n 4. Kampus D (Jl. Margonda Raya Pondok Cina, Depok Phone : 7863819, 7520981,7863788) \n 5. Kampus E (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 6. Kampus G (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 7. Kampus H (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 8. Kampus J (Jl. KH. Noer Ali, Kalimalang Bekasi, Phone : 88860117) \n 9. Kampus K (Jl. Kelapa Dua Raya No.93, Klp. Dua, Kec. Klp. Dua, Kabupaten Tangerang, Banten 15810) \n 10. Kampus L (Jl. Ruko Mutiara Palem Raya Blok C7 No.20, RT.7/RW.14, Cengkareng Tim., Kecamatan Cengkareng, Kota Jakarta Barat, Daerah Khusus Ibukota Jakarta 11730) \n 11. Kampus M (Technopark, Kec. Mande, Kabupaten Cianjur - Jawa Barat) \n 12. Kampus N (Kabupaten Penajam Paser Utara, Kalimantan Timur).",
        "jurusan": "Berikut adalah jurusan yang terdapat pada kampus Universitas Gunadarma : \n- Teknologi Industri : \n (Informatika, Elektro, Mesin, Industri, Agroteknologi). \n- Ilmu Komputer : \n (Sistem Informasi, Sistem Komputer). \n- Ilmu Komunikasi : \n (Komunikasi). \n- Sipil Perencanaan : \n (Arsitektur, Sipil, Desain Interior). \n- Kesehatan dan Farmasi : \n (Farmasi). \n - Psikologi : \n (Psikologi). \n - Kedokteran : \n (Kedokteran). \n - Ekonomi : \n (Akuntansi, Manajemen, Syari'ah). \n - Sastra dan Budaya : \n (Sastra Inggris, Pariwisata, Sastra Tiongkok).",
        "bayaran": "Mengenai info tentang bayaran kampus dapat langsung menghubungi livechat baak dengan link sebagai berikut : https://baak.gunadarma.ac.id/",
        "jadwal": "Untuk jadwal perkuliahan pada kampus Universitas Gunadarma, anda dapat mengakses link yang sudah dicantumkan. Anda dapat mencari jadwal perkuliahan berdasarkan kelas ataupun nama dosen pengajar. Berikut linknya : https://baak.gunadarma.ac.id/jadwal/cariJadKul",
        "kalender": "Untuk melihat kalender akademik kampus universitas gunadarma, anda dapat mengakses link berikut : https://baak.gunadarma.ac.id/downloadAkademik/10",
        "cuti": "Pada dasarnya cuti akademik adalah pembebasan mahasiswa dari kewajiban mengikuti kegiatan akademik selama jangka waktu tertentu. Untuk informasi cuti akademik anda dapat mengakses pada link yang sudah diberikan, berikut linknya : https://baak.gunadarma.ac.id/adminAkademik/2#undefined2",
        "makasih": "Terima kasih kembali karena sudah menggunakan layanan chatbot kampus Universitas Gunadarma. Apakah informasi yang diberikan sudah cukup?",
        "penutup": "Baik, mohon maaf apabila terdapat informasi yang kurang jelas atau kurang dipahami.",
    }
    return responses.get(category, "Maaf, saya tidak mengerti pertanyaan Anda.")

# Fungsi untuk menangani pesan pengguna
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    category = predict_category(text)
    
    if category == "unknown":
        response = "Mohon maaf untuk sekarang bot belum mempelajari hal tersebut."
    else:
        response = get_response(category)
    
    await update.message.reply_text(response)

# Fungsi untuk memulai bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Selamat datang di layanan Chatbot kampus Universitas Gunadarma. Apa yang bisa kami bantu? /help Untuk melihat informasi apa saja yang dapat diberikan oleh Bot.")

# Fungsi untuk bantuan pengguna
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        'Untuk saat ini chatbot UG hanya mampu memberikan informasi mengenai alamat kampus, bayaran kampus, jadwal kampus, kalender akademik kampus, dan cuti semester kampus.'
    )

# Fungsi utama untuk menjalankan bot
def main():
    # Token Telegram Bot Anda
    TOKEN = "7279336549:AAEUcBB_rADIAIQfjDodVcMMQHbQRtrlueA"
    
    # Buat aplikasi bot
    application = Application.builder().token(TOKEN).build()
    
    # Tambahkan handler untuk command /start dan /help
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help))
    
    # Tambahkan handler untuk menangani pesan teks
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Jalankan bot
    application.run_polling()

# Fungsi evaluasi model
def evaluate_model():
    # Data testing
    test_data = [
        ("hai ug", "sapaan"),
        ("halo ug", "sapaan"),
        ("Dimana lokasi kampus?", "alamat"),
        ("Dimana alamat kampus?", "alamat"),
        ("Jurusan gundar ada apa saja", "jurusan"),
        ("Jurusan gundar ada berapa", "jurusan"),
        ("Berapa biaya per semester?", "bayaran"),
        ("Berapa bayaran gundar?", "bayaran"),
        ("Kapan semester dimulai?", "jadwal"),
        ("Jadwal semester gundar", "jadwal"),
        ("Bagaimana kalender gundar", "kalender"),
        ("Kalender akademik gundar", "kalender"),
        ("Cuti semester bisa kapan?", "cuti"),
        ("Cara cuti semester?", "cuti"),
        ("Terima kasih banyak", "makasih"),
        ("Terima kasih informasinya", "makasih"),
        ("Sudah cukup jelas", "penutup"),
        ("Udah jelas", "penutup"),
    ]

    df_test = pd.DataFrame(test_data, columns=["text", "label"])
    X_test = vectorizer.transform(df_test["text"])
    y_test = df_test["label"]

    # Prediksi hasil pada data testing
    y_pred = model.predict(X_test)

    # Lihat akurasi model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Lihat laporan klasifikasi
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Membuat confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()
    # Jalankan evaluasi model untuk melihat performa
    evaluate_model()
