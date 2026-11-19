# Sistem Data Mining Gempa Bumi Sulawesi (tanpa Gorontalo)

Sistem *clustering* K-Means untuk mengelompokkan 225 kejadian gempa (katalog
USGS, Jan 2020 – Jul 2026, M ≥ 4.0) berdasarkan lokasi, kedalaman, dan
magnitudo, lalu menentukan tingkat kerawanan setiap cluster.

## Struktur folder

```
sistem_gempa/
├── app.py                 # Aplikasi web (Streamlit) – 6 menu
├── main.py                # Program command line – seluruh tahapan sekaligus
├── requirements.txt
├── data/
│   └── data_mentah.csv    # Dataset asli Anda
├── src/
│   ├── preprocessing.py   # Pembersihan & rekonstruksi desimal
│   ├── clustering.py      # Normalisasi, K-Means, evaluasi K, profil cluster
│   └── visualisasi.py     # Grafik (matplotlib)
└── output/                # Hasil yang dibuat oleh main.py
```

## Cara menjalankan

```bash
pip install -r requirements.txt

# 1) Versi web (direkomendasikan)
streamlit run app.py          # buka http://localhost:8501

# 2) Versi command line
python main.py                # K otomatis
python main.py --k 4          # K manual
python main.py --log-kedalaman --normalisasi minmax
```

## Tahapan (metodologi KDD)

| Tahap | Isi |
|---|---|
| 1. Selection | Data gempa USGS wilayah Sulawesi Tengah & sekitarnya, tanpa Gorontalo |
| 2. Preprocessing | Buang 274 baris kosong; rekonstruksi tanda desimal yang hilang; validasi koordinat |
| 3. Transformation | Normalisasi Z-score untuk fitur latitude, longitude, kedalaman, magnitudo |
| 4. Data Mining | K-Means (`n_init=20`, `random_state=42`), K = 2..10 dievaluasi |
| 5. Evaluation | Elbow (SSE), Silhouette Score, Davies-Bouldin Index |

### Catatan penting tentang dataset
File CSV kehilangan **semua tanda desimal** (efek format regional Excel
Indonesia): `-164` → `-1.64`, `1215669` → `121.5669`, `45` → `4.5`.
`preprocessing.py` memperbaikinya otomatis:

* Latitude yang ambigu (51 baris) dipilih dengan mencocokkan jarak ke kota acuan
  di kolom `place`. Hasil validasi: **196 gempa yang punya keterangan jarak
  semuanya cocok dalam selisih < 3 km**.
* Kedalaman yang ambigu (25 baris) dipilih yang paling konsisten dengan median
  5 gempa terdekat. Semua koreksi tercatat di kolom `catatan_perbaikan`.
* Kolom pendukung (rms, dmin, horizontalError, depthError, magError) diperbaiki
  dengan aturan rentang nilai wajar — **bersifat perkiraan**, tidak dipakai
  untuk clustering.

Agar 100% akurat, unduh ulang data dari https://earthquake.usgs.gov/earthquakes/search/
lalu buka CSV-nya dengan *Data → From Text/CSV* dan atur pemisah desimal = titik.

## Hasil utama (K = 6, Silhouette 0,3946, DBI 0,7617)

| Cluster | Gempa | Ciri | Kerawanan |
|---|---|---|---|
| 1 | 78 | Dangkal (±12 km), sekitar Poso–Palu, M4.4 | Tinggi |
| 2 | 71 | Dangkal (±22 km), sekitar Luwuk, M4.5 | Sedang |
| 3 | 30 | Dangkal (±25 km), utara Palu (Selat Makassar), M4.5 | Sedang |
| 4 | 25 | Menengah (±169 km), Teluk Tomini (Modisi), M4.4 | Rendah |
| 5 | 20 | Dangkal (±12 km), magnitudo besar M5.5 (maks M6.7) | Tinggi |
| 6 | 1 | Gempa dalam 563 km (outlier) | Rendah |

Elbow menunjukkan siku di K = 5; Silhouette tertinggi di K = 6.

Skor kerawanan = 0,40 × frekuensi + 0,35 × magnitudo rata-rata (keduanya
dinormalisasi min-max antar cluster) + 0,25 × proporsi gempa dangkal (< 70 km).
