"""
SISTEM DATA MINING GEMPA BUMI SULAWESI (tanpa Gorontalo)
=========================================================
Program utama (command line). Menjalankan seluruh tahapan data mining:

    1. Load data CSV mentah
    2. Preprocessing  : buang baris kosong, perbaiki desimal yang hilang,
                        validasi koordinat, fitur turunan
    3. Transformasi   : normalisasi Z-score
    4. Data mining    : K-Means + evaluasi K (Elbow, Silhouette, DBI)
    5. Interpretasi   : profil cluster & tingkat kerawanan
    6. Output         : CSV hasil, grafik PNG, laporan TXT

Cara menjalankan:
    python main.py                                  # pakai data/data_mentah.csv
    python main.py --data file.csv --k 4            # tentukan K manual
    python main.py --fitur latitude longitude kedalaman_km magnitudo
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from clustering import (FITUR_DEFAULT, evaluasi_k, jalankan_kmeans,  # noqa: E402
                        pilih_k_optimal, profil_cluster, siapkan_fitur, titik_siku)
from preprocessing import muat_dan_bersihkan  # noqa: E402
from visualisasi import (grafik_evaluasi, grafik_kedalaman_mag,  # noqa: E402
                         grafik_peta, grafik_tahunan)

def garis(judul=""):
    print("\n" + "=" * 70)
    if judul:
        print(judul)
        print("=" * 70)


def main():
    p = argparse.ArgumentParser(description="Sistem Data Mining Gempa - K-Means")
    p.add_argument("--data", default="data/data_mentah.csv")
    p.add_argument("--k", type=int, default=0, help="jumlah cluster (0 = otomatis)")
    p.add_argument("--fitur", nargs="+", default=FITUR_DEFAULT)
    p.add_argument("--normalisasi", choices=["zscore", "minmax"], default="zscore")
    p.add_argument("--log-kedalaman", action="store_true")
    p.add_argument("--output", default="output")
    a = p.parse_args()
    os.makedirs(a.output, exist_ok=True)
    out = lambda f: os.path.join(a.output, f)  # noqa: E731

    garis("SISTEM DATA MINING GEMPA BUMI SULAWESI - K-MEANS CLUSTERING")

    # 1-2. Load & preprocessing
    data, lap = muat_dan_bersihkan(a.data)
    garis("TAHAP 1-2: LOAD DATA & PREPROCESSING")
    for k, v in lap.items():
        print(f"  {k:<32}: {v}")
    data.to_csv(out("1_data_bersih.csv"), index=False)
    print(f"\n  Data bersih disimpan -> {out('1_data_bersih.csv')}")
    print("\n  Statistik deskriptif fitur:")
    print(data[a.fitur].describe().round(3).to_string())

    # 3-4. Normalisasi & evaluasi K
    garis("TAHAP 3-4: NORMALISASI & PENENTUAN K OPTIMAL")
    X, _ = siapkan_fitur(data, a.fitur, a.normalisasi, a.log_kedalaman)
    ev = evaluasi_k(X, 2, 10)
    k_opt, k_siku = pilih_k_optimal(ev), titik_siku(ev)
    print(ev.round(4).to_string(index=False))
    print(f"\n  K optimal (Silhouette tertinggi) : {k_opt}")
    print(f"  Titik siku (Elbow)              : {k_siku}")
    k = a.k or k_opt
    print(f"  K yang digunakan                : {k}")
    ev.to_csv(out("2_evaluasi_k.csv"), index=False)
    grafik_evaluasi(ev, k_opt, k_siku, out("grafik_1_evaluasi_k.png"))

    # 4. K-Means
    garis(f"TAHAP 4: K-MEANS CLUSTERING (K = {k})")
    hasil, model, X, met = jalankan_kmeans(data, k, a.fitur, a.normalisasi, a.log_kedalaman)
    for kk, v in met.items():
        print(f"  {kk:<16}: {v:.4f}" if isinstance(v, float) else f"  {kk:<16}: {v}")
    hasil.to_csv(out("3_hasil_cluster.csv"), index=False)

    # 5. Profil
    garis("TAHAP 5: PROFIL & INTERPRETASI CLUSTER")
    prof = profil_cluster(hasil)
    prof.to_csv(out("4_profil_cluster.csv"), index=False)
    for _, r in prof.iterrows():
        print(f"\n  CLUSTER {r['cluster']}  -  {r['jumlah_gempa']} gempa ({r['persen']}%)"
              f"  -  Kerawanan: {r['tingkat_kerawanan']} (skor {r['skor_kerawanan']})")
        print(f"    Pusat         : {r['lat_rata2']}, {r['lon_rata2']}")
        print(f"    Kedalaman     : rata2 {r['kedalaman_rata2_km']} km "
              f"({r['kedalaman_min_km']} - {r['kedalaman_maks_km']} km)")
        print(f"    Magnitudo     : rata2 M{r['mag_rata2']}  maks M{r['mag_maks']}")
        print(f"    Wilayah       : {r['wilayah_dominan']}")
        print(f"    Karakteristik : {r['karakteristik']}")

    grafik_peta(hasil, out("grafik_2_peta_cluster.png"))
    grafik_kedalaman_mag(hasil, out("grafik_3_kedalaman_magnitudo.png"))
    grafik_tahunan(hasil, out("grafik_4_gempa_per_tahun.png"))

    # 6. Laporan
    with open(out("5_laporan.txt"), "w", encoding="utf-8") as f:
        f.write("LAPORAN DATA MINING GEMPA BUMI SULAWESI (K-MEANS)\n")
        f.write("=" * 60 + "\n\n[Preprocessing]\n")
        for kk, v in lap.items():
            f.write(f"  {kk}: {v}\n")
        f.write(f"\n[Fitur] {', '.join(a.fitur)}  | Normalisasi: {a.normalisasi}\n")
        f.write("\n[Evaluasi K]\n" + ev.round(4).to_string(index=False) + "\n")
        f.write(f"\nK optimal (Silhouette) = {k_opt}, Elbow = {k_siku}, K dipakai = {k}\n")
        f.write(f"Silhouette = {met['Silhouette']:.4f}, DBI = {met['Davies_Bouldin']:.4f}\n")
        f.write("\n[Profil Cluster]\n" + prof.to_string(index=False) + "\n")
    garis("SELESAI - semua hasil tersimpan di folder '" + a.output + "'")


if __name__ == "__main__":
    main()
