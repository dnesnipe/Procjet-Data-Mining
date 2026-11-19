"""
APLIKASI WEB - SISTEM DATA MINING GEMPA BUMI SULAWESI
======================================================
Jalankan dengan:
    streamlit run app.py

Menu:
    1. Dataset          - upload / lihat data mentah
    2. Preprocessing    - perbaikan desimal & pembersihan data
    3. Eksplorasi Data  - statistik & grafik deskriptif
    4. Penentuan K      - Elbow, Silhouette, Davies-Bouldin
    5. Clustering       - K-Means, peta sebaran, profil & kerawanan
    6. Prediksi Cluster - tentukan cluster untuk data gempa baru
"""

import io
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "src"))
from clustering import (FITUR_DEFAULT, evaluasi_k, jalankan_kmeans,  # noqa: E402
                        pilih_k_optimal, profil_cluster, siapkan_fitur, titik_siku)
from preprocessing import muat_dan_bersihkan  # noqa: E402
from visualisasi import (WARNA, grafik_evaluasi, grafik_kedalaman_mag,  # noqa: E402
                         grafik_peta, grafik_tahunan)

st.set_page_config(page_title="Data Mining Gempa Sulawesi", page_icon="🌏", layout="wide")

DATA_DEFAULT = os.path.join(BASE, "data", "data_mentah.csv")


# ----------------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Membersihkan data...")
def proses_data(isi: bytes):
    return muat_dan_bersihkan(io.BytesIO(isi))


@st.cache_data(show_spinner="Mengevaluasi K...")
def hitung_evaluasi(data, fitur, normalisasi, log_kedalaman, k_maks):
    X, _ = siapkan_fitur(data, fitur, normalisasi, log_kedalaman)
    return evaluasi_k(X, 2, k_maks)


@st.cache_resource(show_spinner="Menjalankan K-Means...")
def hitung_kmeans(data, k, fitur, normalisasi, log_kedalaman):
    return jalankan_kmeans(data, k, list(fitur), normalisasi, log_kedalaman)


def unduh_csv(df, nama, label):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), nama, "text/csv")


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("🌏 Data Mining Gempa")
st.sidebar.caption("K-Means Clustering · Sulawesi (tanpa Gorontalo)")
menu = st.sidebar.radio("Menu", ["1. Dataset", "2. Preprocessing", "3. Eksplorasi Data",
                                 "4. Penentuan K", "5. Clustering", "6. Prediksi Cluster"])

st.sidebar.divider()
st.sidebar.subheader("Sumber data")
upload = st.sidebar.file_uploader("Upload CSV (format USGS)", type=["csv"])
if upload is not None:
    isi = upload.getvalue()
    nama_file = upload.name
elif os.path.exists(DATA_DEFAULT):
    with open(DATA_DEFAULT, "rb") as f:
        isi = f.read()
    nama_file = os.path.basename(DATA_DEFAULT)
else:
    st.warning("Silakan upload file CSV terlebih dahulu.")
    st.stop()

mentah = pd.read_csv(io.BytesIO(isi), dtype=str)
data, laporan = proses_data(isi)

st.sidebar.subheader("Parameter clustering")
fitur = st.sidebar.multiselect("Fitur", FITUR_DEFAULT + ["tahun"], default=FITUR_DEFAULT)
if len(fitur) < 2:
    st.sidebar.error("Pilih minimal 2 fitur.")
    st.stop()
normalisasi = st.sidebar.selectbox("Normalisasi", ["zscore", "minmax"],
                                   format_func=lambda x: {"zscore": "Z-score (StandardScaler)",
                                                          "minmax": "Min-Max (0-1)"}[x])
log_kedalaman = st.sidebar.checkbox("Transformasi log pada kedalaman", value=False)
fitur_t = tuple(fitur)

ev = hitung_evaluasi(data, fitur_t, normalisasi, log_kedalaman, 10)
k_opt, k_siku = pilih_k_optimal(ev), titik_siku(ev)
mode_k = st.sidebar.radio("Jumlah cluster (K)", ["Otomatis (Silhouette)", "Manual"])
k = k_opt if mode_k.startswith("Otomatis") else st.sidebar.slider("K", 2, 10, k_opt)
st.sidebar.info(f"K optimal = **{k_opt}** · Elbow = **{k_siku}** · Dipakai = **{k}**")

hasil, model, X, metrik = hitung_kmeans(data, k, fitur_t, normalisasi, log_kedalaman)
profil = profil_cluster(hasil)


# ----------------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------------
if menu.startswith("1"):
    st.title("1. Dataset Gempa Bumi")
    st.write(f"File: **{nama_file}** — sumber katalog USGS (earthquake.usgs.gov)")
    c = st.columns(4)
    c[0].metric("Baris pada file", len(mentah))
    c[1].metric("Baris berisi data", int(mentah.dropna(how="all").shape[0]))
    c[2].metric("Jumlah kolom", mentah.shape[1])
    c[3].metric("Periode", f"{data['waktu'].min():%Y} – {data['waktu'].max():%Y}")
    st.subheader("Data mentah")
    st.dataframe(mentah.dropna(how="all"), use_container_width=True, height=420)
    st.subheader("Keterangan kolom utama")
    st.table(pd.DataFrame({
        "Kolom": ["time", "latitude", "longitude", "depth", "mag", "magType", "place"],
        "Arti": ["Waktu kejadian (UTC)", "Lintang episentrum", "Bujur episentrum",
                 "Kedalaman hiposentrum (km)", "Magnitudo", "Jenis magnitudo (mb / mww)",
                 "Deskripsi lokasi relatif terhadap kota terdekat"],
    }))

# ----------------------------------------------------------------------------
# 2. Preprocessing
# ----------------------------------------------------------------------------
elif menu.startswith("2"):
    st.title("2. Preprocessing Data")
    st.warning(
        "**Masalah ditemukan:** tanda desimal hilang pada hampir semua kolom angka "
        "(akibat format regional Excel Indonesia). Contoh: latitude `-164` → `-1.64`, "
        "longitude `1215669` → `121.5669`, magnitudo `45` → `4.5`.")
    st.markdown("""
**Langkah preprocessing yang dilakukan sistem:**
1. **Data cleaning** — membuang baris kosong dan duplikat berdasarkan `id`.
2. **Rekonstruksi desimal**
   - *Longitude*: selalu 3 digit bilangan bulat (1xx.xxxx).
   - *Latitude*: maks. 4 desimal & di dalam wilayah studi. Kasus ambigu
     (mis. `-1112` = -1.112 atau -0.1112) diselesaikan dengan mencocokkan jarak
     ke kota acuan pada kolom `place` (mis. "52 km ESE of Palu").
   - *Magnitudo*: format d.d.
   - *Kedalaman*: maks. 3 desimal; kasus ambigu dipilih yang paling konsisten
     dengan median kedalaman 5 gempa terdekat (KNN spasial).
3. **Validasi** koordinat dan **fitur turunan** (tahun, kelas kedalaman, kelas magnitudo, wilayah).
""")
    c = st.columns(5)
    c[0].metric("Baris kosong dibuang", laporan["baris_kosong_dibuang"])
    c[1].metric("Duplikat dibuang", laporan["duplikat_dibuang"])
    c[2].metric("Latitude ambigu", laporan["latitude_ambigu_diselesaikan"])
    c[3].metric("Kedalaman ambigu", laporan["kedalaman_ambigu_diselesaikan"])
    c[4].metric("Data bersih", laporan["baris_bersih"])

    st.subheader("Sebelum vs sesudah")
    contoh = mentah.dropna(how="all").head(len(data))[["latitude", "longitude", "depth", "mag"]]
    banding = pd.concat([contoh.reset_index(drop=True).add_suffix(" (mentah)"),
                         data[["latitude", "longitude", "kedalaman_km", "magnitudo"]]
                         .add_suffix(" (bersih)")], axis=1)
    st.dataframe(banding.head(30), use_container_width=True)

    st.subheader("Data bersih")
    st.dataframe(data, use_container_width=True, height=360)
    unduh_csv(data, "data_bersih.csv", "⬇️ Unduh data bersih (CSV)")

    with st.expander("Log baris yang diperbaiki dengan validasi tambahan"):
        log = data.loc[data["catatan_perbaikan"] != "",
                       ["id", "lokasi", "latitude", "kedalaman_km", "catatan_perbaikan"]]
        st.dataframe(log, use_container_width=True)

# ----------------------------------------------------------------------------
# 3. Eksplorasi
# ----------------------------------------------------------------------------
elif menu.startswith("3"):
    st.title("3. Eksplorasi Data (EDA)")
    c = st.columns(4)
    c[0].metric("Total gempa", len(data))
    c[1].metric("Magnitudo maks", f"M{data['magnitudo'].max():.1f}")
    c[2].metric("Kedalaman rata-rata", f"{data['kedalaman_km'].mean():.1f} km")
    c[3].metric("Gempa dangkal (<70 km)", f"{(data['kedalaman_km'] < 70).mean() * 100:.0f}%")

    st.subheader("Statistik deskriptif")
    st.dataframe(data[["latitude", "longitude", "kedalaman_km", "magnitudo"]]
                 .describe().round(3), use_container_width=True)

    a, b = st.columns(2)
    with a:
        st.subheader("Jumlah gempa per tahun")
        st.pyplot(grafik_tahunan(data))
    with b:
        st.subheader("Kelas kedalaman & magnitudo")
        st.dataframe(data["kelas_kedalaman"].value_counts().rename("jumlah"))
        st.dataframe(data["kelas_magnitudo"].value_counts().rename("jumlah"))

    st.subheader("Gempa per wilayah acuan")
    st.bar_chart(data["wilayah_acuan"].value_counts())
    st.subheader("10 gempa terbesar")
    st.dataframe(data.nlargest(10, "magnitudo")[["waktu", "magnitudo", "kedalaman_km",
                                                 "lokasi"]], use_container_width=True)

# ----------------------------------------------------------------------------
# 4. Penentuan K
# ----------------------------------------------------------------------------
elif menu.startswith("4"):
    st.title("4. Penentuan Jumlah Cluster Optimal")
    st.markdown("""
- **Elbow (SSE)** — cari titik di mana penurunan SSE mulai melandai.
- **Silhouette Score** — kerapatan & keterpisahan cluster (−1 s.d. 1, makin tinggi makin baik).
- **Davies-Bouldin Index** — rasio sebaran dalam cluster terhadap jarak antar cluster (makin rendah makin baik).
""")
    st.pyplot(grafik_evaluasi(ev, k_opt, k_siku))
    st.dataframe(ev.style.highlight_max("Silhouette", color="#cde2fb")
                 .highlight_min("Davies_Bouldin", color="#cde2fb").format(precision=4),
                 use_container_width=True)
    st.success(f"Silhouette tertinggi di **K = {k_opt}** "
               f"({ev.loc[ev.K == k_opt, 'Silhouette'].iloc[0]:.4f}); "
               f"titik siku Elbow di **K = {k_siku}**.")

# ----------------------------------------------------------------------------
# 5. Clustering
# ----------------------------------------------------------------------------
elif menu.startswith("5"):
    st.title(f"5. Hasil K-Means Clustering (K = {k})")
    c = st.columns(4)
    c[0].metric("Jumlah cluster", k)
    c[1].metric("Silhouette", f"{metrik['Silhouette']:.4f}")
    c[2].metric("Davies-Bouldin", f"{metrik['Davies_Bouldin']:.4f}")
    c[3].metric("SSE", f"{metrik['SSE']:.2f}")

    st.subheader("Peta sebaran episentrum")
    peta = hasil.copy()
    peta["warna"] = peta["cluster"].map(lambda c: WARNA[(c - 1) % len(WARNA)])
    peta["ukuran"] = (peta["magnitudo"] - 3.5) * 2500
    st.map(peta, latitude="latitude", longitude="longitude", color="warna", size="ukuran")
    st.caption(" · ".join(f"<span style='color:{WARNA[(c - 1) % len(WARNA)]}'>●</span> "
                          f"Cluster {c}" for c in sorted(hasil["cluster"].unique())),
               unsafe_allow_html=True)

    a, b = st.columns(2)
    with a:
        st.pyplot(grafik_peta(hasil))
    with b:
        st.pyplot(grafik_kedalaman_mag(hasil))

    st.subheader("Profil & tingkat kerawanan cluster")
    st.dataframe(profil, use_container_width=True, hide_index=True)
    for _, r in profil.iterrows():
        ikon = {"Tinggi": "🔴", "Sedang": "🟠", "Rendah": "🟢"}[r["tingkat_kerawanan"]]
        st.markdown(f"{ikon} **Cluster {r['cluster']}** — {r['jumlah_gempa']} gempa, "
                    f"kerawanan **{r['tingkat_kerawanan']}**. {r['karakteristik']}.")
    st.caption("Skor kerawanan = 0,40 × frekuensi (ternormalisasi) + 0,35 × magnitudo "
               "rata-rata (ternormalisasi) + 0,25 × proporsi gempa dangkal. "
               "Tinggi > 0,65 · Sedang 0,40–0,65 · Rendah < 0,40.")

    st.subheader("Data hasil clustering")
    pilih = st.multiselect("Filter cluster", sorted(hasil["cluster"].unique()))
    tampil = hasil[hasil["cluster"].isin(pilih)] if pilih else hasil
    st.dataframe(tampil.drop(columns=["catatan_perbaikan"]), use_container_width=True,
                 height=360)
    a, b = st.columns(2)
    with a:
        unduh_csv(hasil, "hasil_cluster.csv", "⬇️ Unduh hasil clustering")
    with b:
        unduh_csv(profil, "profil_cluster.csv", "⬇️ Unduh profil cluster")

# ----------------------------------------------------------------------------
# 6. Prediksi
# ----------------------------------------------------------------------------
elif menu.startswith("6"):
    st.title("6. Prediksi Cluster Gempa Baru")
    st.write("Masukkan parameter gempa baru; sistem menentukan cluster terdekat "
             "berdasarkan model K-Means yang sudah dilatih.")
    nilai = {}
    kol = st.columns(len(fitur))
    batas = {"latitude": (-3.5, 2.5, -1.0), "longitude": (118.0, 126.0, 120.5),
             "kedalaman_km": (0.0, 700.0, 10.0), "magnitudo": (3.0, 9.0, 5.0),
             "tahun": (2000.0, 2030.0, 2026.0)}
    for i, f in enumerate(fitur):
        lo, hi, dv = batas.get(f, (float(data[f].min()), float(data[f].max()),
                                   float(data[f].median())))
        nilai[f] = kol[i].number_input(f, min_value=lo, max_value=hi, value=dv,
                                       step=0.1 if f != "tahun" else 1.0)
    if st.button("🔍 Prediksi cluster", type="primary"):
        baru = pd.DataFrame([nilai])
        # gunakan scaler dari data latih agar skala konsisten
        _, scaler = siapkan_fitur(data, fitur, normalisasi, log_kedalaman)
        vb = baru[list(fitur)].astype(float).copy()
        if log_kedalaman and "kedalaman_km" in vb:
            vb["kedalaman_km"] = np.log1p(vb["kedalaman_km"])
        label_asli = model.predict(scaler.transform(vb.to_numpy()))[0]
        # petakan ke nomor cluster yang ditampilkan
        peta_label = (pd.DataFrame({"asli": model.labels_, "tampil": hasil["cluster"]})
                      .drop_duplicates().set_index("asli")["tampil"])
        c = int(peta_label[label_asli])
        r = profil[profil["cluster"] == c].iloc[0]
        st.success(f"Gempa ini masuk **Cluster {c}** — tingkat kerawanan "
                   f"**{r['tingkat_kerawanan']}**")
        st.write(r["karakteristik"] + ".")
        st.dataframe(profil[profil["cluster"] == c], hide_index=True,
                     use_container_width=True)

st.sidebar.divider()
st.sidebar.caption("Metode: KDD — Selection → Preprocessing → Transformation → "
                   "Data Mining (K-Means) → Interpretation/Evaluation")
