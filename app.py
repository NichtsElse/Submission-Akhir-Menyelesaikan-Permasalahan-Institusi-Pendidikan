import streamlit as st
import pandas as pd
import joblib

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Deteksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="centered"
)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('model_do_prediction.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Kamus Data (Mapping)
# Dictionary ini digunakan untuk mengubah angka menjadi teks yang mudah dibaca di UI
marital_map = {
    1: "Single (Belum Menikah)", 
    2: "Married (Menikah)", 
    3: "Widower (Duda/Janda)", 
    4: "Divorced (Bercerai)", 
    5: "Facto Union (Kumpul Kebo)", 
    6: "Legally Separated (Berpisah Hukum)"
}

# Mapping untuk Application Mode (Beberapa jalur umum dari dataset asli)
app_mode_map = {
    1: "1st phase - general contingent",
    15: "International student (Mahasiswa Internasional)",
    17: "2nd phase - general contingent",
    39: "Over 23 years old (Jalur usia di atas 23 tahun)",
    43: "Transfer student (Mahasiswa Pindahan)",
    44: "Technological specialization diploma",
    51: "Change of course (Pindah Jurusan)"
    # Jika ada angka lain di luar ini, kita tangani dengan fallback di format_func
}

yes_no_map = {1: "Ya", 0: "Tidak"}
gender_map = {1: "Laki-laki", 0: "Perempuan"}
tuition_map = {1: "Lancar (Tidak Menunggak)", 0: "Menunggak"}

# 4. Header Aplikasi
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")
st.markdown("Pilih data mahasiswa pada masing-masing kategori di bawah ini, lalu klik tombol **Proses Prediksi**.")
st.markdown("---")

# 5. Membuat Tabs untuk Kategori Input
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "👤 Profil", 
    "🏫 Latar Belakang", 
    "💰 Finansial", 
    "📊 Semester 1", 
    "📈 Semester 2"
])

# Kategori 1: Profil & Demografi
with tab1:
    st.subheader("Data Demografi")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender (Jenis Kelamin)", options=[1, 0], format_func=lambda x: gender_map[x])
        age = st.number_input("Age at Enrollment (Usia Masuk)", min_value=15, max_value=80, value=20)
    with col2:
        marital = st.selectbox("Marital Status (Status Pernikahan)", options=list(marital_map.keys()), format_func=lambda x: marital_map[x])
        displaced = st.selectbox("Displaced (Mahasiswa Pendatang)", options=[1, 0], format_func=lambda x: yes_no_map[x])

# Kategori 2: Latar Belakang Akademik
with tab2:
    st.subheader("Riwayat Pendaftaran")
    
    # Karena Application Mode ada banyak, kita beri opsi umum dan membiarkan user input jika tidak ada
    opsi_app_mode = list(app_mode_map.keys()) + [0] # 0 sebagai opsi 'Lainnya'
    app_mode_select = st.selectbox(
        "Application Mode (Jalur Masuk)", 
        options=opsi_app_mode, 
        format_func=lambda x: app_mode_map.get(x, "Jalur Lainnya (Input Manual)")
    )
    
    # Jika memilih lainnya, munculkan input angka manual
    if app_mode_select == 0:
        app_mode = st.number_input("Masukkan Kode Jalur Masuk:", min_value=1, value=1)
    else:
        app_mode = app_mode_select

    col1, col2 = st.columns(2)
    with col1:
        prev_grade = st.number_input("Previous Grade (Nilai Kelulusan Sebelumnya)", min_value=0.0, max_value=200.0, value=120.0, step=1.0)
    with col2:
        adm_grade = st.number_input("Admission Grade (Nilai Ujian Masuk)", min_value=0.0, max_value=200.0, value=120.0, step=1.0)

# Kategori 3: Finansial
with tab3:
    st.subheader("Status Finansial & Ekonomi")
    col1, col2 = st.columns(2)
    with col1:
        scholarship = st.selectbox("Scholarship Holder (Penerima Beasiswa)", options=[1, 0], format_func=lambda x: yes_no_map[x])
        debtor = st.selectbox("Debtor (Memiliki Hutang)", options=[1, 0], format_func=lambda x: yes_no_map[x])
    with col2:
        tuition = st.selectbox("Tuition Fees Up to Date (Pembayaran UKT)", options=[1, 0], format_func=lambda x: tuition_map[x])

# Kategori 4: Kinerja Semester 1
with tab4:
    st.subheader("Performa Akademik - Semester 1")
    col1, col2 = st.columns(2)
    with col1:
        s1_enrolled = st.number_input("Curricular Units 1st Sem Enrolled (SKS Diambil)", value=6, min_value=0)
        s1_approved = st.number_input("Curricular Units 1st Sem Approved (SKS Lulus)", value=5, min_value=0)
    with col2:
        s1_grade = st.number_input("Curricular Units 1st Sem Grade (Rata-rata Nilai/IP)", value=12.0, step=0.1, min_value=0.0)

# Kategori 5: Kinerja Semester 2
with tab5:
    st.subheader("Performa Akademik - Semester 2")
    col1, col2 = st.columns(2)
    with col1:
        s2_enrolled = st.number_input("Curricular Units 2nd Sem Enrolled (SKS Diambil)", value=6, min_value=0)
        s2_eval = st.number_input("Curricular Units 2nd Sem Evaluations (Evaluasi Diikuti)", value=6, min_value=0)
        s2_no_eval = st.number_input("Curricular Units 2nd Sem Without Eval (Tanpa Evaluasi)", value=0, min_value=0)
    with col2:
        s2_approved = st.number_input("Curricular Units 2nd Sem Approved (SKS Lulus)", value=5, min_value=0)
        s2_grade = st.number_input("Curricular Units 2nd Sem Grade (Rata-rata Nilai/IP)", value=12.0, step=0.1, min_value=0.0)

st.markdown("---")

# 6. Tombol Proses & Logika Prediksi
if st.button("📊 Proses Prediksi", use_container_width=True):
    # Menyusun data sesuai dengan urutan fitur yang diminta model
    input_features = {
        'Age_at_enrollment': age,
        'Debtor': debtor,
        'Gender': gender,
        'Application_mode': app_mode,
        'Curricular_units_2nd_sem_without_evaluations': s2_no_eval,
        'Marital_status': marital,
        'Previous_qualification_grade': prev_grade,
        'Curricular_units_2nd_sem_evaluations': s2_eval,
        'Displaced': displaced,
        'Admission_grade': adm_grade,
        'Curricular_units_1st_sem_enrolled': s1_enrolled,
        'Curricular_units_2nd_sem_enrolled': s2_enrolled,
        'Scholarship_holder': scholarship,
        'Tuition_fees_up_to_date': tuition,
        'Curricular_units_1st_sem_grade': s1_grade,
        'Curricular_units_1st_sem_approved': s1_approved,
        'Curricular_units_2nd_sem_grade': s2_grade,
        'Curricular_units_2nd_sem_approved': s2_approved
    }
    
    df_input = pd.DataFrame([input_features])
    
    # Prediksi
    prediction = model.predict(df_input)
    prediction_proba = model.predict_proba(df_input)

    st.markdown("---")
    st.subheader("💡 Hasil Analisis:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **STATUS: BERISIKO TINGGI (DROPOUT)**")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][1]*100:.2f}%**")
        st.info("Rekomendasi: Segera jadwalkan sesi bimbingan konseling dan akademik untuk mahasiswa ini.")
    else:
        st.success(f"✅ **STATUS: AMAN (DIPREDIKSI LULUS/GRADUATE)**")
        st.write(f"Tingkat Keyakinan Model: **{prediction_proba[0][0]*100:.2f}%**")

st.markdown("---")
st.caption("Proyek Data Science - Jaya Jaya Institut")
