import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42
from langdetect.lang_detect_exception import LangDetectException
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from streamlit_option_menu import option_menu

# ---------------------------------------------------------
# SETUP HALAMAN STREAMLIT
# ---------------------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Bawang Putih Herbal - Naive Bayes",
    page_icon="🧄",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS (TEMA: CLEAN MODERN MINIMALIST WITH ORANGE ACCENT)
# ---------------------------------------------------------
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #1A1A1A;
    }

    /* Background Aplikasi */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #EAEAEA;
    }

    /* Primary Buttons */
    div.stButton > button[kind="primary"], .stDownloadButton > button {
        background-color: #E85D04 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        box-shadow: 0 4px 12px rgba(232, 93, 4, 0.2);
        transition: all 0.3s ease;
    }
    
    div.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
        background-color: #DC2F02 !important;
        box-shadow: 0 6px 16px rgba(220, 47, 2, 0.3);
        transform: translateY(-1px);
    }

    /* Secondary Buttons */
    div.stButton > button[kind="secondary"] {
        background-color: #F4F4F5 !important;
        color: #1A1A1A !important;
        border: 1px solid #E4E4E7 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button[kind="secondary"]:hover {
        background-color: #E4E4E7 !important;
    }

    /* Inputs & Textareas */
    .stTextArea textarea, .stSelectbox select, .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid #E4E4E7 !important;
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #E85D04 !important;
        box-shadow: 0 0 0 2px rgba(232, 93, 4, 0.15) !important;
    }

  
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #EAEAEA !important;
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    font-weight: 600;
    color: #71717A !important;
}

div[data-testid="stMetricValue"] {
    font-weight: 800;
    color: #1A1A1A !important;
}

/* Memastikan teks markdown umum tidak ikut gelap */
p, span, label, .stMarkdown {
    color: #1A1A1A !important;
}

    /* Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #71717A;
        background-color: #F4F4F5;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #E85D04 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# INISIALISASI SESSION STATE
# ---------------------------------------------------------
if 'single_result' not in st.session_state:
    st.session_state['single_result'] = None

if 'batch_result' not in st.session_state:
    st.session_state['batch_result'] = None

if 'batch_vec' not in st.session_state:
    st.session_state['batch_vec'] = None

if 'batch_col_name' not in st.session_state:
    st.session_state['batch_col_name'] = None

# ---------------------------------------------------------
# LOAD MODEL & VECTORIZER DARI FILE .PKL
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    with open('model_nb.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_ml_assets()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Gagal memuat file model (.pkl). Pastikan 'model_nb.pkl' dan 'tfidf.pkl' berada di folder yang sama!\nError: {e}")

# ---------------------------------------------------------
# NLP TOOLS & STOPWORDS
# ---------------------------------------------------------
@st.cache_resource
def load_nlp_tools():
    stemmer = StemmerFactory().create_stemmer()
    factory_stop = StopWordRemoverFactory()
    stopwords = factory_stop.get_stop_words()
    # Pengecualian kata negasi agar sentimen tidak terdistorsi
    negasi = ['tidak', 'bukan', 'kurang']
    stopwords = [w for w in stopwords if w not in negasi]
    return stemmer, stopwords

stemmer, custom_stopwords = load_nlp_tools()

# ---------------------------------------------------------
# FUNGSI PREPROCESSING (SAMA DENGAN NOTEBOOK COLAB)
# ---------------------------------------------------------
def cleaning(text):
    text = str(text)
    # Filter Bahasa Indonesia menggunakan langdetect
    words = text.split()
    if len(words) >= 5:
            try:
                lang = detect(text)
                if lang not in ['id']:
                    return ""
            except LangDetectException:
                pass # Lanjutkan proses jika gagal deteksi

    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'â\w+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\bga\b', 'tidak', text)
    text = re.sub(r'\bgak\b', 'tidak', text)
    text = text.replace("ngefek", "efek")
    text = text.replace("nyembuhin", "sembuh")
    return text

def remove_stopwords(tokens):
    return [word for word in tokens if word not in custom_stopwords]

def stem_text(tokens):
    return stemmer.stem(' '.join(tokens)).split()

def preprocess_pipeline(raw_text):
    """
    Fungsi terpadu untuk pengujian input teks tunggal (Single Prediction)
    """
    text = cleaning(raw_text)
    if not text:
        return ""
    text = normalize(text)
    tokens = text.split()  # Tokenizing
    tokens = remove_stopwords(tokens)
    tokens = stem_text(tokens)
    return ' '.join(tokens)  # Final Text

# ---------------------------------------------------------
# HELPER: TAMPILAN LABEL SENTIMEN
# ---------------------------------------------------------
LABEL_STYLE = {
    "positif": ("POSITIF 😊", "success"),
    "negatif": ("NEGATIF 😡", "error"),
    "netral":  ("NETRAL 😐", "warning"),
}

def show_prediction(label):
    text, kind = LABEL_STYLE.get(label, (label.upper(), "info"))
    getattr(st, kind)(f"### Sentimen: {text}")

def show_confidence(model, vec_input):
    if not hasattr(model, "predict_proba"):
        return
    proba = model.predict_proba(vec_input)[0]
    classes = model.classes_
    st.write("**Tingkat keyakinan model:**")
    order = sorted(zip(classes, proba), key=lambda x: -x[1])
    for cls, p in order:
        st.write(f"{cls.capitalize()}")
        st.progress(float(p), text=f"{p*100:.1f}%")

# ---------------------------------------------------------
# HEADER APLIKASI
# ---------------------------------------------------------
st.markdown("""
    <div style="padding: 20px 0 10px 0;">
        <p style="color: #E85D04; font-weight: 700; font-size: 0.9em; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 4px;">
            Sistem Analisis Sentimen
        </p>
        <h1 style="font-weight: 800; font-size: 2.3em; color: #1A1A1A; margin-top: 0;">
            Opini Herbal Bawang Putih
        </h1>
        <p style="color: #71717A; font-size: 1.05em; max-width: 700px; margin-bottom: 25px;">
            Platform klasifikasi sentimen masyarakat terhadap pemanfaatan bawang putih sebagai obat herbal di sosial media menggunakan algoritma Naive Bayes.
        </p>
    </div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR DESIGN
# =========================================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: left; padding: 10px 0 5px 0;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 32px;">🧄</span>
                <h3 style="color: #1A1A1A; margin: 0; font-weight: 800; font-size: 1.3em;">
                    GARLIC HERBS
                </h3>
            </div>
            <p style="color: #71717A; font-size: 0.8em; margin-top: 4px; font-weight: 500;">
                Sentiment Analysis Platform
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()

    menu = option_menu(
        menu_title="NAVIGASI",
        options=["Uji Kalimat Tunggal", "Analisis Dataset", "Tentang Penelitian"],
        icons=["chat-text", "table", "info-circle"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "0px!important", "background-color": "transparent"},
            "icon": {"color": "#E85D04", "font-size": "15px"}, 
            "nav-link": {
                "font-size": "14px", 
                "text-align": "left", 
                "margin": "4px 0px", 
                "border-radius": "8px",
                "color": "#1A1A1A",
                "font-weight": "600",
                "--hover-color": "#F4F4F5"
            },
            "nav-link-selected": {
                "background-color": "#1A1A1A", 
                "color": "#FFFFFF",
                "font-weight": "700"
            },
        }
    )

    st.divider()

    st.markdown("""
        <div style="text-align: left; font-size: 0.75em; color: #A1A1AA; margin-top: 10px;">
            © 2026 Putra Rangga P.<br>Teknik Informatika - UNSERA
        </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 1: UJI KALIMAT TUNGGAL
# ---------------------------------------------------------
if menu == "Uji Kalimat Tunggal":
    st.subheader("Uji Sentimen Kalimat / Tweet")

    user_input = st.text_area(
        "Masukkan teks tweet/opini mengenai bawang putih herbal:",
        value="",
        placeholder="Contoh: Bawang putih ampuh banget buat nurunkan kolesterol dan hipertensi!"
    )

    if st.button("Analisis Sentimen", type="primary"):
        if user_input.strip() != "":
            if not model_loaded:
                st.error("Model belum berhasil dimuat. Periksa kembali file .pkl Anda.")
            else:
                processed_res = preprocess_pipeline(user_input)
                vec_input = vectorizer.transform([processed_res])
                prediction = model.predict(vec_input)[0]

                st.session_state['single_result'] = {
                    'raw_text': user_input,
                    'processed_res': processed_res,
                    'vec_input': vec_input,
                    'prediction': prediction
                }
        else:
            st.warning("Harap masukkan teks terlebih dahulu!")

    if st.session_state['single_result'] is not None:
        res = st.session_state['single_result']
        st.write("---")
        
        st.success("**Hasil Preprocessing (Final Text):**")
        st.write(res['processed_res'] if res['processed_res'] else "_(kosong setelah preprocessing)_")

        st.divider()
        st.subheader("📌 Hasil Klasifikasi Sentimen:")
        show_prediction(str(res['prediction']).lower())
        show_confidence(model, res['vec_input'])

# ---------------------------------------------------------
# MENU 2: BATCH ANALYSIS (UPLOAD CSV)
# ---------------------------------------------------------
elif menu == "Analisis Dataset":
    st.subheader("📂 Analisis Dataset Sentimen")
    uploaded_file = st.file_uploader("Unggah file CSV data tweet:", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Cek jumlah baris murni vs nama kolom


        if st.session_state['batch_result'] is None:
            st.write(f"**Preview Data Terunggah ({len(df)} baris):**")
            st.dataframe(df, use_container_width=True)

            col_name = st.selectbox("Pilih kolom teks tweet:", df.columns)

            if st.button("Proses Seluruh Data", type="primary"):
                if not model_loaded:
                    st.error("Model/Vectorizer belum berhasil dimuat. Periksa kembali file .pkl Anda.")
                else:
                    with st.spinner("Memproses pipeline preprocessing, TF-IDF, & prediksi model... Mohon tunggu..."):
                        raw_total_count = len(df)
                        # 1. Cleaning Teks & Normalisasi (Termasuk filter langdetect)
                        df['clean_text'] = df[col_name].astype(str).apply(cleaning).apply(normalize)
                        
                        # 2. Hapus teks kosong akibat filter langdetect / pembersihan
                        df = df[df['clean_text'].str.strip() != ''].copy()
                        
                        # 3. Otomatis Hapus Duplikat berdasarkan teks yang sudah bersih
                        df = df.drop_duplicates(subset=['clean_text'], keep='first').reset_index(drop=True)
                        
                        # 4. Tokenizing (Persis seperti notebook: lambda x: x.split())
                        df['tokens'] = df['clean_text'].apply(lambda x: x.split())
                        
                        # 5. Stopword Removal (Menjaga kata negasi: tidak, bukan, kurang)
                        df['tokens'] = df['tokens'].apply(remove_stopwords)
                        
                        # 6. Stemming Sastrawi
                        df['tokens'] = df['tokens'].apply(stem_text)
                        
                        # 7. Final Text untuk Vektorisasi TF-IDF
                        df['final_text'] = df['tokens'].apply(lambda x: ' '.join(x))
                        
                        # 8. Transformasi TF-IDF & Prediksi Model Naive Bayes
                        vec_batch = vectorizer.transform(df['final_text'])
                        df['prediksi_sentimen'] = model.predict(vec_batch)

                        # Simpan ke Session State
                        st.session_state['batch_result'] = df
                        st.session_state['batch_vec'] = vec_batch
                        st.session_state['batch_col_name'] = col_name
                        st.session_state['raw_total'] = raw_total_count

                    st.success("Selesai memproses seluruh data!")
                    st.rerun()

        else:
            if st.button("🔄 Reset / Upload File Baru", type="secondary"):
                st.session_state['batch_result'] = None
                st.session_state['batch_vec'] = None
                st.session_state['batch_col_name'] = None
                st.rerun()

    if st.session_state['batch_result'] is not None:
        df_res = st.session_state['batch_result']
        vec_b = st.session_state['batch_vec']
        c_name = st.session_state['batch_col_name']
        total_raw = st.session_state['raw_total']

        st.divider()
        st.subheader("📊 Ringkasan Hasil Prediksi")
        
        sentimen_counts = df_res['prediksi_sentimen'].value_counts().to_dict()
        valid_total = len(df_res)
        pos_val = sentimen_counts.get('positif', 0)
        net_val = sentimen_counts.get('netral', 0)
        neg_val = sentimen_counts.get('negatif', 0)

        # Menggunakan total_raw untuk Total Raw Data
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Raw Data", total_raw, f"{valid_total} Data Unik")
        m2.metric("Positif", pos_val, f"{(pos_val/valid_total*100):.1f}%")
        m3.metric("Netral", net_val, f"{(net_val/valid_total*100):.1f}%")
        m4.metric("Negatif", neg_val, f"{(neg_val/valid_total*100):.1f}%")

        st.divider()

        tab1, tab2, tab3 = st.tabs(["📋 Hasil Preprocessing & Prediksi", "🔠 Hasil Matriks TF-IDF", "📈 Visualisasi Grafik"])

        with tab1:
            st.markdown("#### Tabel Preprocessing Teks & Hasil Prediksi")
            # Menampilkan tahapan lebih lengkap agar terlihat progres preprocessing-nya
            cols_to_display = [c_name, 'clean_text', 'final_text', 'prediksi_sentimen']
            st.dataframe(df_res[cols_to_display], use_container_width=True)

        with tab2:
            st.markdown("#### Tabel Ekstraksi Fitur (Matriks TF-IDF)")
            st.caption("Menampilkan bobot nilai TF-IDF untuk setiap kata/fitur pada 100 data pertama.")
            
            feature_names = vectorizer.get_feature_names_out()
            tfidf_df = pd.DataFrame(
                vec_b[:100].toarray(), 
                columns=feature_names
            )
            non_zero_cols = tfidf_df.loc[:, (tfidf_df != 0).any(axis=0)]
            st.dataframe(non_zero_cols, use_container_width=True)

        with tab3:
            st.markdown("#### Visualisasi Distribusi Sentimen")
            col_g1, col_g2 = st.columns(2)
            chart_data = df_res['prediksi_sentimen'].value_counts()

            custom_colors = ['#E85D04', '#27272A', '#A1A1AA']

            with col_g1:
                st.markdown("**Distribusi Jumlah Sentimen**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x=chart_data.index, y=chart_data.values, ax=ax, palette=custom_colors)
                ax.set_ylabel("Jumlah Tweet")
                sns.despine()
                st.pyplot(fig)

            with col_g2:
                st.markdown("**Proporsi Sentimen (%)**")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.pie(chart_data.values, labels=chart_data.index, autopct='%1.1f%%',
                        startangle=90, colors=custom_colors[:len(chart_data)])
                st.pyplot(fig2)

        st.divider()
        st.download_button(
            "⬇️ Unduh Hasil Prediksi Lengkap (CSV)",
            data=df_res.to_csv(index=False).encode('utf-8'),
            file_name="hasil_analisis_sentimen_lengkap.csv",
            mime="text/csv"
        )
# ---------------------------------------------------------
# MENU 3: TENTANG PENELITIAN
# ---------------------------------------------------------
elif menu == "Tentang Penelitian":
    st.subheader("ℹ️ Informasi Skripsi")
    
    st.markdown("""
        <div style="
            background-color: #FFFFFF;
            border: 1px solid #EAEAEA;
            border-left: 5px solid #E85D04;
            padding: 20px 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
            margin-bottom: 25px;
        ">
            <h4 style="margin-top: 0; color: #1A1A1A; font-weight: 700; font-size: 1.1em;">
                Analisis Sentimen Masyarakat Terhadap Pemanfaatan Bawang Putih Sebagai Obat Herbal Di Media Sosial X Menggunakan Metode Naive Bayes
            </h4>
            <hr style="border: 0; border-top: 1px solid #F4F4F5; margin: 15px 0;">
            <div style="display: grid; grid-template-columns: 120px 1fr; gap: 8px; font-size: 0.95em; color: #3F3F46;">
                <strong>Nama</strong> <span>: Putra Rangga Purnama</span>
                <strong>NIM</strong> <span>: 11222109</span>
                <strong>Program Studi</strong> <span>: Teknik Informatika</span>
                <strong>Instansi</strong> <span>: Universitas Serang Raya (UNSERA)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.divider()
    
    st.subheader("📊 Performa Model")
    
    m_col, _ = st.columns([1, 1])
    with m_col:
        st.metric("Akurasi Model", "74%")
    
    st.caption("SMOTE diterapkan untuk menangani ketidakseimbangan kelas pada data training.")