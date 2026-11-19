"""
Modul Clustering - Sistem Data Mining Gempa Bumi Sulawesi
==========================================================

Metode : K-Means Clustering
Fitur  : latitude, longitude, kedalaman_km, magnitudo (dapat dipilih)
Tahapan:
    1. Normalisasi fitur (Z-score / StandardScaler) agar satuan berbeda
       (derajat, km, skala Richter) memiliki bobot yang setara.
    2. Penentuan jumlah cluster optimal (K) dengan:
         - Elbow Method   (SSE / inertia)
         - Silhouette Score (semakin tinggi semakin baik, rentang -1..1)
         - Davies-Bouldin Index (semakin rendah semakin baik)
    3. K-Means dengan K terpilih.
    4. Profil tiap cluster + tingkat kerawanan (Tinggi / Sedang / Rendah).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

FITUR_DEFAULT = ["latitude", "longitude", "kedalaman_km", "magnitudo"]
SEED = 42


def siapkan_fitur(df: pd.DataFrame, fitur=FITUR_DEFAULT, metode="zscore",
                  log_kedalaman=False):
    """Kembalikan matriks fitur yang sudah dinormalisasi + scaler-nya."""
    X = df[list(fitur)].astype(float).copy()
    if log_kedalaman and "kedalaman_km" in X:
        X["kedalaman_km"] = np.log1p(X["kedalaman_km"])
    scaler = StandardScaler() if metode == "zscore" else MinMaxScaler()
    return scaler.fit_transform(X.to_numpy()), scaler


def evaluasi_k(X: np.ndarray, k_min=2, k_max=10) -> pd.DataFrame:
    """Hitung SSE, Silhouette, dan DBI untuk setiap K."""
    baris = []
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
        baris.append({
            "K": k,
            "SSE": km.inertia_,
            "Silhouette": silhouette_score(X, km.labels_),
            "Davies_Bouldin": davies_bouldin_score(X, km.labels_),
        })
    return pd.DataFrame(baris)


def pilih_k_optimal(evaluasi: pd.DataFrame) -> int:
    """K optimal = Silhouette tertinggi (DBI sebagai pemutus seri)."""
    urut = evaluasi.sort_values(["Silhouette", "Davies_Bouldin"], ascending=[False, True])
    return int(urut.iloc[0]["K"])


def titik_siku(evaluasi: pd.DataFrame) -> int:
    """Titik siku (elbow) dengan metode jarak maksimum ke garis ujung-ujung."""
    x = evaluasi["K"].to_numpy(float)
    y = evaluasi["SSE"].to_numpy(float)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    # jarak ke garis dari titik pertama ke terakhir
    jarak = np.abs(yn[0] + (yn[-1] - yn[0]) * xn - yn) / np.sqrt(1 + (yn[-1] - yn[0]) ** 2)
    return int(x[np.argmax(jarak)])


def jalankan_kmeans(df: pd.DataFrame, k: int, fitur=FITUR_DEFAULT,
                    metode="zscore", log_kedalaman=False):
    """Jalankan K-Means, kembalikan (df_berlabel, model, X, metrik)."""
    X, scaler = siapkan_fitur(df, fitur, metode, log_kedalaman)
    model = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)

    # Urutkan nomor cluster berdasarkan jumlah anggota (Cluster 1 = terbesar)
    urutan = pd.Series(model.labels_).value_counts().index.tolist()
    peta = {lama: baru + 1 for baru, lama in enumerate(urutan)}
    hasil = df.copy()
    hasil["cluster"] = [peta[c] for c in model.labels_]

    metrik = {
        "K": k,
        "SSE": float(model.inertia_),
        "Silhouette": float(silhouette_score(X, model.labels_)) if k > 1 else np.nan,
        "Davies_Bouldin": float(davies_bouldin_score(X, model.labels_)) if k > 1 else np.nan,
        "iterasi": int(model.n_iter_),
    }
    return hasil, model, X, metrik


def profil_cluster(hasil: pd.DataFrame) -> pd.DataFrame:
    """Ringkasan statistik setiap cluster + tingkat kerawanan."""
    g = hasil.groupby("cluster")
    prof = pd.DataFrame({
        "jumlah_gempa": g.size(),
        "persen": (g.size() / len(hasil) * 100).round(1),
        "lat_rata2": g["latitude"].mean().round(3),
        "lon_rata2": g["longitude"].mean().round(3),
        "kedalaman_rata2_km": g["kedalaman_km"].mean().round(1),
        "kedalaman_min_km": g["kedalaman_km"].min().round(1),
        "kedalaman_maks_km": g["kedalaman_km"].max().round(1),
        "mag_rata2": g["magnitudo"].mean().round(2),
        "mag_maks": g["magnitudo"].max(),
        "persen_dangkal": (g["kedalaman_km"].apply(lambda s: (s < 70).mean()) * 100).round(1),
        "wilayah_dominan": g["wilayah_acuan"].agg(lambda s: ", ".join(
            f"{w} ({n})" for w, n in s.value_counts().head(3).items())),
    })

    # Skor kerawanan (0-1): gabungan frekuensi, magnitudo, dan kedangkalan.
    def norm(s):
        return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0 + 0.5

    prof["skor_kerawanan"] = (0.4 * norm(prof["jumlah_gempa"])
                              + 0.35 * norm(prof["mag_rata2"])
                              + 0.25 * prof["persen_dangkal"] / 100).round(3)
    prof["tingkat_kerawanan"] = pd.cut(prof["skor_kerawanan"], [-0.01, 0.4, 0.65, 1.01],
                                       labels=["Rendah", "Sedang", "Tinggi"]).astype(str)
    prof["karakteristik"] = prof.apply(_deskripsi, axis=1)
    return prof.reset_index()


def _deskripsi(r) -> str:
    if r["kedalaman_rata2_km"] < 70:
        jenis = "gempa dangkal (sesar/kerak bumi)"
    elif r["kedalaman_rata2_km"] < 300:
        jenis = "gempa menengah (lempeng subduksi)"
    else:
        jenis = "gempa dalam (lempeng subduksi)"
    wilayah = r["wilayah_dominan"].split(" (")[0]
    return (f"Didominasi {jenis} di sekitar {wilayah}, "
            f"magnitudo rata-rata M{r['mag_rata2']:.1f}")
