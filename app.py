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
    # Pastikan nama file sesuai dengan file .pkl Anda
    return joblib.load('model_do_prediction.pkl.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Judul Aplikasi
st.title("🎓 Sistem Prediksi Dropout Mahasiswa")
st.write("Masukkan data mahasiswa di bawah ini untuk melihat prediksi status kelulusan.")

st.markdown("---")

# 4. Input Form (Dibagi menjadi 2 kolom agar rapi)
col1, col2 = st.columns(2)

with col1:
    st.subheader("Profil & Demografi")
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Laki-laki" if x == 1 else "Perempuan")
    age = st.number_input("Age at Enrollment", min_value=15, max_value=80, value=20)
    marital = st.selectbox("Marital Status", options=[1, 2, 3, 4, 5, 6], help="1: Single, 2: Married, dst.")
    displaced = st.selectbox("Displaced (Pindahan)", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    
    st.subheader("Status Finansial")
    scholarship = st.selectbox("Scholarship Holder", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    debtor = st.selectbox("Debtor (Punya Hutang)", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    tuition = st.selectbox("Tuition Fees Up to Date", options=[0, 1], format_func=lambda x: "Lancar" if x == 1 else "Menunggak")

with col2:
    st.subheader("Akademik Input")
    app_mode = st.number_input("Application Mode (Kode)", value=1)
    prev_grade = st.number_input("Previous Qualification Grade", value=120.0)
    adm_grade = st.number_input("Admission Grade", value=120.0)
    
    st.subheader("Kinerja Semester 1")
    s1_enrolled = st.number_input("Units 1st Sem Enrolled", value=6)
    s1_approved = st.number_input("Units 1st Sem Approved", value=5)
    s1_grade = st.number_input("Units 1st Sem Grade", value=12.0)

    st.subheader("Kinerja Semester 2")
    s2_enrolled = st.number_input("Units 2nd Sem Enrolled", value=6)
    s2_eval = st.number_input("Units 2nd Sem Evaluations", value=6)
    s2_approved = st.number_input("Units 2nd Sem Approved", value=5)
    s2_grade = st.number_input("Units 2nd Sem Grade", value=12.0)
    s2_no_eval = st.number_input("Units 2nd Sem Without Eval", value=0)

# 5. Proses Prediksi
if st.button("Proses Prediksi", use_container_width=True):
    # Menyusun data sesuai urutan fitur yang diminta
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
    st.subheader("Hasil Analisis:")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **STATUS: BERISIKO DROPOUT**")
        st.write(f"Tingkat Keyakinan Model: {prediction_proba[0][1]*100:.2f}%")
        st.warning("Rekomendasi: Segera jadwalkan sesi bimbingan konseling dan akademik untuk mahasiswa ini.")
    else:
        st.success(f"✅ **STATUS: DIPREDIKSI LULUS (GRADUATE)**")
        st.write(f"Tingkat Keyakinan Model: {prediction_proba[0][0]*100:.2f}%")

st.markdown("---")
st.caption("Jaya Jaya Institut - Project Data Science")
