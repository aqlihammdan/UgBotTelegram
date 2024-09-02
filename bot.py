import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import random
import pandas as pd
import logging
import numpy as np
import re

# Logging untuk debugging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Unduh data punkt untuk tokenisasi NLTK
nltk.download('punkt')

# Data untuk melatih model
data = [
    ("Dimana letak kampus gundar?", "alamat"),
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

# Convert to DataFrame for better readability and manipulation
df = pd.DataFrame(data, columns=["text", "label"])

# Memisahkan data dan label
texts, labels = df["text"], df["label"]

# Membuat dan melatih model
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)
logger.info("Model has been trained.")

# Fungsi untuk melakukan preprocessing pada input
def preprocess_input(text):
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

# Fungsi prediksi dengan pemeriksaan probabilitas, panjang input, dan input tidak jelas
def predict_category(text, threshold=0.24):
    global model
    try:
        # Preprocessing input
        text = preprocess_input(text)

        # Periksa apakah input tidak jelas (gibberish)
        if is_gibberish(text):
            logger.info("Detected gibberish input, setting category to 'unknown'")
            return "unknown"

        # Periksa panjang input, jika terlalu pendek, langsung kembalikan 'unknown'
        if len(text) < 3:
            logger.info("Input too short, setting category to 'unknown'")
            return "unknown"
        
        probabilities = model.predict_proba([text])[0]
        max_prob = np.max(probabilities)
        category = model.predict([text])[0]
        
        # Jika probabilitas tertinggi di bawah ambang batas, setel kategori sebagai 'unknown'
        if max_prob < threshold:
            logger.info(f"Low confidence ({max_prob:.2f}), setting category to 'unknown'")
            return "unknown"
        
        logger.info(f"Predicted category: {category} with probability {max_prob:.2f}")
        return category
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "unknown"


# Jawaban untuk setiap kategori
responses = {
    "alamat": [
        "Untuk lokasi Universitas Gunadarma terletak di berbagai wilayah : \n 1. Kampus A (Jl. Kenari nomor 13 Jakarta Pusat, 10430 Phone : 330220, 330226) \n 2. Kampus B (Jl. Salemba Bluntas Jakarta Pusat) \n 3. Kampus C (Jl. Salemba Raya nomor 53 Jakarta Pusat Phone : 3906518, 3908568 Fax : 3100325) \n 4. Kampus D (Jl. Margonda Raya Pondok Cina, Depok Phone : 7863819, 7520981,7863788) \n 5. Kampus E (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 6. Kampus G (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 7. Kampus H (Jl. Akses Kelapa Dua Kelapa Dua, Cimanggis Phone : 8719525, 8710561, 8727541 ext. 103,106 Fax : 8710561) \n 8. Kampus J (Jl. KH. Noer Ali, Kalimalang Bekasi, Phone : 88860117) \n 9. Kampus K (Jl. Kelapa Dua Raya No.93, Klp. Dua, Kec. Klp. Dua, Kabupaten Tangerang, Banten 15810) \n 10. Kampus L (Jl. Ruko Mutiara Palem Raya Blok C7 No.20, RT.7/RW.14, Cengkareng Tim., Kecamatan Cengkareng, Kota Jakarta Barat, Daerah Khusus Ibukota Jakarta 11730) \n 11. Kampus M (Technopark, Kec. Mande, Kabupaten Cianjur - Jawa Barat) \n 12. Kampus N (Kabupaten Penajam Paser Utara, Kalimantan Timur)"
    ],
    "jurusan": [
        "Berikut adalah jurusan yang terdapat pada kampus Universitas Gunadarma : \n- Teknologi Industri : \n (Informatika, Elektro, Mesin, Industri, Agroteknologi). \n- Ilmu Komputer : \n (Sistem Informasi, Sistem Komputer). \n- Ilmu Komunikasi : \n (Komunikasi). \n- Sipil Perencanaan : \n (Arsitektur, Sipil, Desain Interior). \n- Kesehatan dan Farmasi : \n (Farmasi). \n - Psikologi : \n (Psikologi). \n - Kedokteran : \n (Kedokteran). \n - Ekonomi : \n (Akuntansi, Manajemen, Syari'ah). \n - Sastra dan Budaya : \n (Sastra Inggris, Pariwisata, Sastra Tiongkok)."
    ],
    "bayaran": [
        "Mengenai info tentang bayaran kampus dapat langsung menghubungi livechat baak dengan link sebagai berikut : https://baak.gunadarma.ac.id/"
    ],
    "jadwal": [
        "Untuk jadwal perkuliahan pada kampus Universitas Gunadarma, anda dapat mengakses link yang sudah dicantumkan. Anda dapat mencari jadwal perkuliahan berdasarkan kelas ataupun nama dosen pengajar. Berikut linknya : https://baak.gunadarma.ac.id/jadwal/cariJadKul"
    ],
    "kalender": [
        "Untuk melihat kalender akademik kampus universitas gunadarma, anda dapat mengakses link berikut : https://baak.gunadarma.ac.id/downloadAkademik/9"
    ],
    "cuti": [
        "Pada dasarnya cuti akademik adalah pembebasan mahasiswa dari kewajiban mengikuti kegiatan akademik selama jangka waktu tertentu. Untuk informasi cuti akademik anda dapat mengakses pada link yang sudah diberikan, berikut linknya : https://baak.gunadarma.ac.id/adminAkademik/2#undefined2"
    ],
    "makasih": [
        "Terima kasih kembali karena sudah menggunakan layanan chatbot kampus Universitas Gunadarma. Apakah informasi yang diberikan sudah cukup?"
    ],
    "penutup": [
        "Baik, mohon maaf apabila terdapat informasi yang kurang jelas atau kurang dipahami."
    ],
    "unknown": [
        "Mohon maaf untuk sekarang bot belum mempelajari hal tersebut."
    ]
}

# Token API dari BotFather
TOKEN = '7279336549:AAEUcBB_rADIAIQfjDodVcMMQHbQRtrlueA'

# Fungsi untuk memulai bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Selamat datang di layanan Chatbot kampus Universitas Gunadarma. Apa yang bisa kami bantu? /help Untuk melihat informasi apa saja yang dapat diberikan oleh Bot.')

# Fungsi untuk bantuan pengguna
async def help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Untuk saat ini chatbot UG hanya mampu memberikan informasi mengenai alamat kampus, bayaran kampus, jadwal kampus, kalender akademik kampus, dan cuti semester kampus.')

# Fungsi untuk menangani pesan
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        text = update.message.text
        logger.info(f"Received message: {text}")
        
        category = predict_category(text)
        logger.info(f"Message category: {category}")

        # Explicit check to ensure category is valid
        if category not in responses:
            logger.info("Category not in responses, setting to 'unknown'")
            category = "unknown"
        
        response = random.choice(responses[category])
        logger.info(f"Responding with: {response}")
        
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error handling message: {e}")
        await update.message.reply_text("Sorry, something went wrong. Please try again later.")

def main():
    try:
        application = Application.builder().token(TOKEN).build()

        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help))
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        logger.info("Bot is starting...")
        application.run_polling()
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == '__main__':
    main()
