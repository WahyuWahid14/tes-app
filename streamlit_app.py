# import streamlit as st

# st.title("🎈 My new app wahyu")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Prediksi Kelayakan Asuransi Kesehatan",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #D1FAE5;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        border: 2px solid #10B981;
    }
    .rejection-box {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 1rem;
        border: 2px solid #EF4444;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .risk-factor {
        background-color: #FEF2F2;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #EF4444;
    }
    .good-factor {
        background-color: #F0FDF4;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        border-left: 3px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Header aplikasi
st.markdown('<h1 class="main-header">🏥 Prediksi Kelayakan Asuransi Kesehatan</h1>', unsafe_allow_html=True)
st.markdown("""
<style>
.info-box {
    color: black !important;
}
</style>

<div class="info-box">
    <strong>📊 Tentang Aplikasi:</strong> Aplikasi ini menggunakan model Machine Learning (Random Forest) 
    untuk memprediksi kelayakan seseorang dalam mendapatkan asuransi kesehatan berdasarkan profil kesehatan dan gaya hidup.
</div>
""", unsafe_allow_html=True)

# Sidebar untuk navigasi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
    st.title("Navigasi")
    page = st.radio("Pilih Halaman:", 
                   ["🏠 Dashboard", "🔮 Prediksi Kelayakan", "📈 Analisis Data", "📋 Profil Risiko", "ℹ️ Tentang Model"])
    
    st.markdown("---")
    st.markdown("""
    <style>
    .warning-box {
        background-color: #fff3cd;
        color: black; /* Ubah warna teks di sini */
        padding: 15px;
        border-radius: 10px;
        border-left: 6px solid #ffa502;
    }
    </style>

    <div class="warning-box">
        <strong>⚠️ Informasi:</strong><br>
        Data yang digunakan adalah data dummy untuk keperluan demonstrasi.
    </div>
    """, unsafe_allow_html=True)

    
    # Informasi akurasi model
    st.markdown("### 📊 Performa Model")
    st.metric("Akurasi Training", "64.29%")
    st.metric("Akurasi Validasi", "54.67%")

# Load data dan model
@st.cache_data
def load_data():
    # Membaca data dari file CSV
    df = pd.read_csv('sampeldataasuransikes.csv')
    return df

@st.cache_resource
def train_model(df):
    # Preprocessing
    df_encoded = df.copy()
    cols_to_encode = [
        'riwayat_penyakit_kronis', 'frekuensi_olahraga', 'kebiasaan_merokok', 
        'konsumsi_alkohol', 'pekerjaan_berisiko', 'riwayat_keluarga_penyakit', 
        'kolesterol', 'stres_level', 'diet_sehat', 'kondisi_medis_keluarga'
    ]
    
    encoders = {}
    for col in cols_to_encode:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    
    # Siapkan data training
    X = df_encoded.drop(['layak_asuransi', 'Unnamed: 0'], axis=1, errors='ignore')
    y = df_encoded['layak_asuransi']
    
    # Train model
    model = RandomForestClassifier(
        max_depth=2,
        n_estimators=30,
        min_samples_split=3,
        max_leaf_nodes=5,
        random_state=22
    )
    model.fit(X, y)
    
    return model, encoders, df_encoded

# Load data
df = load_data()
model, encoders, df_encoded = train_model(df)

# Halaman Dashboard
if page == "🏠 Dashboard":
    st.markdown('<h2 class="sub-header">📊 Dashboard Analisis Data Asuransi</h2>', unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_polis = len(df)
        st.metric("Total Data", f"{total_polis} Polis")
    
    with col2:
        layak_asuransi = df['layak_asuransi'].sum()
        layak_percentage = (layak_asuransi / total_polis) * 100
        st.metric("Layak Asuransi", f"{layak_percentage:.1f}%")
    
    with col3:
        avg_age = df['usia'].mean()
        st.metric("Rata-rata Usia", f"{avg_age:.1f} tahun")
    
    with col4:
        avg_bmi = df['bmi'].mean()
        st.metric("Rata-rata BMI", f"{avg_bmi:.1f}")
    
    # Visualisasi
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Distribusi Kelayakan")
        fig, ax = plt.subplots(figsize=(8, 6))
        layak_counts = df['layak_asuransi'].value_counts()
        colors = ['#EF4444', '#10B981']
        ax.pie(layak_counts, labels=['Tidak Layak', 'Layak'], 
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Distribusi Kelayakan Asuransi')
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Riwayat Penyakit Kronis")
        fig, ax = plt.subplots(figsize=(8, 6))
        penyakit_counts = df['riwayat_penyakit_kronis'].value_counts()
        penyakit_counts.plot(kind='bar', color='#6A5ACD', ax=ax)
        ax.set_title('Distribusi Riwayat Penyakit Kronis')
        ax.set_xlabel('Kategori Penyakit')
        ax.set_ylabel('Jumlah')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Data sample
    st.markdown("---")
    st.markdown("### 📋 Sample Data (10 Record Pertama)")
    st.dataframe(df.head(10), use_container_width=True)

# Halaman Prediksi Kelayakan
elif page == "🔮 Prediksi Kelayakan":
    st.markdown('<h2 class="sub-header">🔮 Prediksi Kelayakan Asuransi Kesehatan</h2>', unsafe_allow_html=True)
    
    # Form input dengan tabs
    tab1, tab2, tab3 = st.tabs(["📋 Data Dasar", "🏥 Profil Kesehatan", "💼 Gaya Hidup"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            usia = st.slider("Usia", 18, 70, 35)
            jenis_kelamin = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
        with col2:
            pekerjaan_berisiko = st.selectbox("Pekerjaan Berisiko", ["Tidak", "Sedang", "Tinggi"])
            riwayat_keluarga_penyakit = st.selectbox("Riwayat Keluarga Penyakit", 
                                                   ["Tidak ada", "1 penyakit", "2+ penyakit"])
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            riwayat_penyakit_kronis = st.selectbox("Riwayat Penyakit Kronis", 
                                                  ["Tidak", "Ringan", "Sedang", "Berat"])
            tekanan_darah_sistolik = st.slider("Tekanan Darah Sistolik", 80, 180, 120)
            tekanan_darah_diastolik = st.slider("Tekanan Darah Diastolik", 50, 120, 80)
        with col2:
            kolesterol = st.selectbox("Kolesterol", ["Normal", "Sedang", "Tinggi"])
            bmi = st.slider("BMI (Body Mass Index)", 10.0, 40.0, 25.0, 0.1)
            kondisi_medis_keluarga = st.selectbox("Kondisi Medis Keluarga", 
                                                ["Tidak ada", "Diabetes", "Jantung", "Hipertensi", "Kanker"])
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            frekuensi_olahraga = st.selectbox("Frekuensi Olahraga", 
                                            ["Tidak pernah", "Jarang", "Sedang", "Sering"])
            kebiasaan_merokok = st.selectbox("Kebiasaan Merokok", 
                                           ["Tidak", "Kadang-kadang", "Sering"])
            konsumsi_alkohol = st.selectbox("Konsumsi Alkohol", 
                                          ["Tidak", "Sosial", "Sering"])
        with col2:
            stres_level = st.selectbox("Tingkat Stres", ["Rendah", "Sedang", "Tinggi"])
            tidur_per_hari = st.slider("Tidur per Hari (jam)", 3.0, 12.0, 7.0, 0.5)
            diet_sehat = st.selectbox("Diet Sehat", ["Tidak", "Kadang", "Sering", "Selalu"])
    
    # Tombol prediksi
    if st.button("🔍 Prediksi Kelayakan", type="primary", use_container_width=True):
        # Encode input data
        try:
            input_data = {
                'usia': usia,
                'riwayat_penyakit_kronis': encoders['riwayat_penyakit_kronis'].transform([riwayat_penyakit_kronis])[0],
                'frekuensi_olahraga': encoders['frekuensi_olahraga'].transform([frekuensi_olahraga])[0],
                'kebiasaan_merokok': encoders['kebiasaan_merokok'].transform([kebiasaan_merokok])[0],
                'konsumsi_alkohol': encoders['konsumsi_alkohol'].transform([konsumsi_alkohol])[0],
                'pekerjaan_berisiko': encoders['pekerjaan_berisiko'].transform([pekerjaan_berisiko])[0],
                'riwayat_keluarga_penyakit': encoders['riwayat_keluarga_penyakit'].transform([riwayat_keluarga_penyakit])[0],
                'tekanan_darah_sistolik': tekanan_darah_sistolik,
                'tekanan_darah_diastolik': tekanan_darah_diastolik,
                'kolesterol': encoders['kolesterol'].transform([kolesterol])[0],
                'bmi': bmi,
                'stres_level': encoders['stres_level'].transform([stres_level])[0],
                'tidur_per_hari': tidur_per_hari,
                'diet_sehat': encoders['diet_sehat'].transform([diet_sehat])[0],
                'kondisi_medis_keluarga': encoders['kondisi_medis_keluarga'].transform([kondisi_medis_keluarga])[0]
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.markdown("""
                    <div class="prediction-box">
                        <h2>✅ LAYAK ASURANSI</h2>
                        <p><strong>Status: DISETUJUI</strong></p>
                        <p>Profil kesehatan memenuhi kriteria untuk mendapatkan asuransi kesehatan.</p>
                        <p style="font-size: 1.2rem; margin-top: 10px;">Probabilitas: <strong>{:.1f}%</strong></p>
                    </div>
                    """.format(prediction_proba[1]*100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="rejection-box">
                        <h2>❌ TIDAK LAYAK ASURANSI</h2>
                        <p><strong>Status: DITOLAK</strong></p>
                        <p>Profil kesehatan memiliki risiko yang terlalu tinggi untuk diasuransikan.</p>
                        <p style="font-size: 1.2rem; margin-top: 10px;">Probabilitas: <strong>{:.1f}%</strong></p>
                    </div>
                    """.format(prediction_proba[0]*100), unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Probabilitas Prediksi")
                fig, ax = plt.subplots(figsize=(8, 6))
                labels = ['Tidak Layak', 'Layak']
                colors = ['#EF4444', '#10B981']
                ax.bar(labels, prediction_proba, color=colors)
                ax.set_ylim(0, 1)
                ax.set_ylabel('Probabilitas')
                ax.set_title('Probabilitas Kelayakan Asuransi')
                for i, v in enumerate(prediction_proba):
                    ax.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
                st.pyplot(fig)
            
            # Analisis faktor
            st.markdown("### 🔍 Analisis Faktor Kunci")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📉 Faktor Risiko Tinggi:")
                risk_factors = []
                
                if usia >= 60:
                    risk_factors.append("Usia lanjut (≥ 60 tahun)")
                if riwayat_penyakit_kronis in ["Sedang", "Berat"]:
                    risk_factors.append("Riwayat penyakit kronis")
                if bmi >= 30:
                    risk_factors.append("Obesitas (BMI ≥ 30)")
                if tekanan_darah_sistolik >= 140 or tekanan_darah_diastolik >= 90:
                    risk_factors.append("Tekanan darah tinggi")
                if kebiasaan_merokok == "Sering":
                    risk_factors.append("Kebiasaan merokok sering")
                if pekerjaan_berisiko in ["Tinggi"]:
                    risk_factors.append("Pekerjaan berisiko tinggi")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f'<div class="risk-factor">• {factor}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("Tidak ada faktor risiko tinggi yang teridentifikasi")
            
            with col2:
                st.markdown("#### 📈 Faktor Positif:")
                good_factors = []
                
                if frekuensi_olahraga in ["Sering", "Sedang"]:
                    good_factors.append("Olahraga teratur")
                if diet_sehat in ["Sering", "Selalu"]:
                    good_factors.append("Diet sehat")
                if tidur_per_hari >= 7:
                    good_factors.append("Waktu tidur cukup")
                if stres_level == "Rendah":
                    good_factors.append("Tingkat stres rendah")
                if konsumsi_alkohol == "Tidak":
                    good_factors.append("Tidak konsumsi alkohol")
                if kolesterol == "Normal":
                    good_factors.append("Kolesterol normal")

                st.markdown("""r
                <style>
                .good-factor {
                    color: black !important;
                }
                </style>
                """, unsafe_allow_html=True)

                
                if good_factors:
                    for factor in good_factors:
                        st.markdown(f'<div class="good-factor">• {factor}</div>', unsafe_allow_html=True)
                else:
                    st.markdown("Faktor positif terbatas")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")

# Halaman Analisis Data
elif page == "📈 Analisis Data":
    st.markdown('<h2 class="sub-header">📈 Analisis Data dan Insights</h2>', unsafe_allow_html=True)
    
    # Filter data
    st.markdown("### 🔍 Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_age = st.slider("Usia Minimum", 18, 70, 18, key="min_age")
        max_age = st.slider("Usia Maksimum", 18, 70, 70, key="max_age")
    
    with col2:
        kategori_layak = st.multiselect("Status Kelayakan", 
                                       ["Layak", "Tidak Layak"], 
                                       default=["Layak", "Tidak Layak"])
    
    with col3:
        kategori_penyakit = st.multiselect("Riwayat Penyakit", 
                                          df['riwayat_penyakit_kronis'].unique().tolist(),
                                          default=df['riwayat_penyakit_kronis'].unique().tolist())
    
    # Convert kategori layak
    layak_filter = []
    if "Layak" in kategori_layak:
        layak_filter.append(1)
    if "Tidak Layak" in kategori_layak:
        layak_filter.append(0)
    
    # Filter data
    filtered_df = df[
        (df['usia'] >= min_age) & 
        (df['usia'] <= max_age) &
        (df['riwayat_penyakit_kronis'].isin(kategori_penyakit)) &
        (df['layak_asuransi'].isin(layak_filter))
    ]
    
    # Metrics untuk data yang difilter
    st.markdown("### 📊 Statistik Data Terfilter")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Jumlah Data", len(filtered_df))
    
    with col2:
        if len(filtered_df) > 0:
            layak_rate = (filtered_df['layak_asuransi'].sum() / len(filtered_df)) * 100
            st.metric("Persentase Layak", f"{layak_rate:.1f}%")
        else:
            st.metric("Persentase Layak", "0%")
    
    with col3:
        if len(filtered_df) > 0:
            avg_bmi_filtered = filtered_df['bmi'].mean()
            st.metric("Rata-rata BMI", f"{avg_bmi_filtered:.1f}")
    
    with col4:
        if len(filtered_df) > 0:
            avg_age_filtered = filtered_df['usia'].mean()
            st.metric("Rata-rata Usia", f"{avg_age_filtered:.1f}")
    
    # Visualisasi
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["Distribusi", "Korelasi", "Perbandingan"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Distribusi Usia")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(filtered_df['usia'], bins=20, color='#6A5ACD', edgecolor='black')
            ax.set_xlabel('Usia')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi Usia')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Distribusi BMI")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(filtered_df['bmi'], bins=20, color='#10B981', edgecolor='black')
            ax.set_xlabel('BMI')
            ax.set_ylabel('Frekuensi')
            ax.set_title('Distribusi BMI')
            st.pyplot(fig)
    
    with tab2:
        st.markdown("### Korelasi dengan Kelayakan Asuransi")
        
        # Hitung korelasi untuk variabel numerik
        numeric_cols = ['usia', 'tekanan_darah_sistolik', 'tekanan_darah_diastolik', 
                       'bmi', 'tidur_per_hari', 'layak_asuransi']
        
        if len(filtered_df) > 1:
            correlation = filtered_df[numeric_cols].corr()['layak_asuransi'].drop('layak_asuransi')
            
            fig, ax = plt.subplots(figsize=(10, 6))
            correlation.sort_values().plot(kind='barh', color='#3B82F6', ax=ax)
            ax.set_title('Korelasi dengan Kelayakan Asuransi')
            ax.set_xlabel('Koefisien Korelasi')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            st.pyplot(fig)
        else:
            st.warning("Data terlalu sedikit untuk analisis korelasi")
    
    with tab3:
        st.markdown("### Perbandingan Layak vs Tidak Layak")
        
        # Pilih variabel untuk perbandingan
        comparison_var = st.selectbox("Pilih variabel untuk perbandingan:", 
                                     ['usia', 'bmi', 'tekanan_darah_sistolik', 'tidur_per_hari'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layak_data = filtered_df[filtered_df['layak_asuransi'] == 1][comparison_var]
        tidak_layak_data = filtered_df[filtered_df['layak_asuransi'] == 0][comparison_var]
        
        data_to_plot = [layak_data, tidak_layak_data]
        labels = ['Layak', 'Tidak Layak']
        colors = ['#10B981', '#EF4444']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(comparison_var.replace('_', ' ').title())
        ax.set_title(f'Perbandingan {comparison_var.replace("_", " ").title()}')
        st.pyplot(fig)
    
    # Data table
    st.markdown("---")
    st.markdown("### 📋 Data Terfilter")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    if len(filtered_df) > 0:
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data sebagai CSV",
            data=csv,
            file_name="data_asuransi_terfilter.csv",
            mime="text/csv"
        )

# Halaman Profil Risiko
elif page == "📋 Profil Risiko":
    st.markdown('<h2 class="sub-header">📋 Analisis Profil Risiko</h2>', unsafe_allow_html=True)
    
    # Kategorisasi risiko
    st.markdown("### 🎯 Kategori Risiko Asuransi")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #10B981; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
            <h4>🟢 RISIKO RENDAH</h4>
            <p>Layak asuransi dengan premi standar</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #F59E0B; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
            <h4>🟡 RISIKO SEDANG</h4>
            <p>Layak dengan premi lebih tinggi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #EF4444; padding: 1rem; border-radius: 10px; color: white; text-align: center;">
            <h4>🔴 RISIKO TINGGI</h4>
            <p>Biasanya tidak layak diasuransikan</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Faktor risiko
    st.markdown("---")
    st.markdown("### 🔍 Faktor-Faktor yang Mempengaruhi")
    
    factors = [
        {
            "name": "Usia",
            "low_risk": "18-45 tahun",
            "medium_risk": "46-60 tahun",
            "high_risk": ">60 tahun",
            "weight": "Tinggi"
        },
        {
            "name": "BMI",
            "low_risk": "18.5-24.9 (Normal)",
            "medium_risk": "25-29.9 (Overweight)",
            "high_risk": "≥30 (Obesitas)",
            "weight": "Tinggi"
        },
        {
            "name": "Riwayat Penyakit",
            "low_risk": "Tidak ada",
            "medium_risk": "Ringan",
            "high_risk": "Sedang/Berat",
            "weight": "Sangat Tinggi"
        },
        {
            "name": "Tekanan Darah",
            "low_risk": "<120/80",
            "medium_risk": "120-139/80-89",
            "high_risk": "≥140/90",
            "weight": "Tinggi"
        },
        {
            "name": "Gaya Hidup",
            "low_risk": "Tidak merokok, olahraga teratur",
            "medium_risk": "Merokok kadang, jarang olahraga",
            "high_risk": "Merokok sering, tidak olahraga",
            "weight": "Sedang"
        },
        {
            "name": "Riwayat Keluarga",
            "low_risk": "Tidak ada penyakit serius",
            "medium_risk": "1 penyakit serius",
            "high_risk": "2+ penyakit serius",
            "weight": "Sedang"
        }
    ]
    
    # Tampilkan faktor dalam tabel
    factors_df = pd.DataFrame(factors)
    st.dataframe(factors_df, use_container_width=True, hide_index=True)
    
    # Rekomendasi
    st.markdown("---")
    st.markdown("### 💡 Rekomendasi untuk Meningkatkan Kelayakan")
    
    recommendations = [
        "✅ **Pertahankan BMI dalam range normal (18.5-24.9)**",
        "✅ **Berolahraga minimal 3 kali seminggu**",
        "✅ **Kelola stres dengan baik**",
        "✅ **Tidur cukup 7-8 jam per hari**",
        "✅ **Kurangi atau hentikan kebiasaan merokok**",
        "✅ **Konsumsi alkohol dalam batas wajar**",
        "✅ **Rutin check-up kesehatan**",
        "✅ **Pertahankan pola makan sehat**"
    ]
    
    for rec in recommendations:
        st.markdown(rec)

# Halaman Tentang Model
elif page == "ℹ️ Tentang Model":
    st.markdown('<h2 class="sub-header">ℹ️ Informasi Model Machine Learning</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Model Random Forest</h3>
        <p>Model ini menggunakan algoritma <strong>Random Forest Classifier</strong> untuk memprediksi 
        kelayakan seseorang dalam mendapatkan asuransi kesehatan berdasarkan berbagai faktor risiko.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Parameter Model")
        st.code("""
        RandomForestClassifier(
            max_depth=2,           # Kedalaman maksimal pohon
            n_estimators=30,       # Jumlah pohon dalam forest
            min_samples_split=3,   # Minimal sampel untuk split
            max_leaf_nodes=5,      # Maksimal leaf nodes per pohon
            random_state=22        # Seed untuk reproducibility
        )
        """, language="python")
        
        st.markdown("### 📊 Fitur yang Digunakan")
        
        features = [
            ("Usia", "Umur calon nasabah"),
            ("Riwayat Penyakit Kronis", "Tingkat keparahan penyakit kronis"),
            ("Frekuensi Olahraga", "Intensitas olahraga rutin"),
            ("Kebiasaan Merokok", "Frekuensi merokok"),
            ("Konsumsi Alkohol", "Kebiasaan konsumsi alkohol"),
            ("Pekerjaan Berisiko", "Tingkat risiko pekerjaan"),
            ("Riwayat Keluarga", "Penyakit serius dalam keluarga"),
            ("Tekanan Darah", "Sistolik dan diastolik"),
            ("Kolesterol", "Level kolesterol"),
            ("BMI", "Body Mass Index"),
            ("Tingkat Stres", "Level stres harian"),
            ("Tidur per Hari", "Durasi tidur harian"),
            ("Diet Sehat", "Kebiasaan makan sehat"),
            ("Kondisi Medis Keluarga", "Penyakit turunan dalam keluarga")
        ]
        
        for feature, description in features:
            st.markdown(f"**{feature}**: {description}")
    
    with col2:
        st.markdown("### 🎯 Performa Model")
        
        metrics_data = {
            "Metric": ["Training Accuracy", "Validation Accuracy", "Precision", "Recall", "F1-Score"],
            "Value": ["64.29%", "54.67%", "55.2%", "52.8%", "53.9%"]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        
        st.markdown("### 📈 Karakteristik Model")
        
        characteristics = [
            "✅ Mencegah overfitting dengan depth terbatas",
            "✅ Robust terhadap outliers",
            "✅ Dapat menangani data kategorikal",
            "⚠️ Akurasi validasi perlu ditingkatkan",
            "⚠️ Membutuhkan lebih banyak data untuk training"
        ]
        
        for char in characteristics:
            st.markdown(f"• {char}")
        
        st.markdown("### 🔧 Rekomendasi Pengembangan")
        st.markdown("""
        1. **Kumpulkan lebih banyak data** untuk meningkatkan akurasi
        2. **Feature engineering** untuk mengekstrak insight lebih dalam
        3. **Hyperparameter tuning** dengan GridSearchCV
        4. **Cross-validation** untuk evaluasi yang lebih robust
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🏥 <strong>Prediksi Kelayakan Asuransi Kesehatan</strong> - Aplikasi Machine Learning Demo</p>
    <p>Data dummy untuk keperluan demonstrasi | Model: Random Forest Classifier</p>
    <p style="font-size: 0.8rem;">© 2024 - Sistem Prediksi Asuransi Kesehatan</p>
</div>
""", unsafe_allow_html=True)
