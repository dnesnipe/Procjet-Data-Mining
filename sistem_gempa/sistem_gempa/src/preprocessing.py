"""
Modul Preprocessing - Sistem Data Mining Gempa Bumi Sulawesi
=============================================================

Masalah pada dataset:
    File CSV hasil ekspor Excel (pengaturan regional Indonesia) kehilangan
    tanda desimal. Contoh:
        latitude  -164      -> seharusnya -1.64
        longitude 1215669   -> seharusnya 121.5669
        mag       45        -> seharusnya 4.5
        depth     135283    -> seharusnya 135.283

Modul ini merekonstruksi tanda desimal secara otomatis dengan aturan
berbasis format data USGS dan validasi geografis:

    1. Longitude  : wilayah Sulawesi selalu 1xx.xxxx  -> 3 digit bilangan bulat.
    2. Latitude   : maksimal 4 angka desimal (format USGS) dan berada di
                    rentang wilayah studi. Jika masih ambigu (mis. -1112 bisa
                    -1.112 atau -0.1112), dipilih kandidat yang paling sesuai
                    dengan kolom 'place' (mis. "52 km ESE of Palu").
    3. Magnitudo  : selalu d.d  (45 -> 4.5, 5 -> 5.0).
    4. Kedalaman  : maksimal 3 angka desimal. Nilai 1-2 digit dianggap bulat
                    (10 km = kedalaman tetap USGS), 5-6 digit dibagi 1000.
                    Nilai 3-4 digit ambigu -> dipilih kandidat yang paling dekat
                    dengan median kedalaman 5 gempa terdekat (imputasi KNN).
    5. Kolom pendukung (rms, dmin, horizontalError, depthError, magError)
       diperbaiki dengan aturan rentang nilai yang wajar.

Setiap baris yang diperbaiki dengan tebakan diberi tanda pada kolom
'catatan_perbaikan' agar bisa dicek ulang ke katalog USGS.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

# Koordinat kota acuan (perkiraan) yang dipakai USGS pada kolom 'place'.
# Kota yang tidak ada di sini akan diestimasi otomatis dari data.
KOTA_ACUAN = {
    "Palu": (-0.8950, 119.8594),
    "Poso": (-1.3962, 120.7525),
    "Luwuk": (-0.9516, 122.7875),
    "Masamba": (-2.5530, 120.3270),
    "Mamuju": (-2.6748, 118.8885),
}

ARAH = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

BATAS_LAT = (-3.5, 2.5)        # rentang lintang wilayah studi
BATAS_LON = (118.0, 126.0)     # rentang bujur wilayah studi

KOLOM_UTAMA = ["time", "latitude", "longitude", "depth", "mag", "magType", "place"]


# ----------------------------------------------------------------------------
# Utilitas
# ----------------------------------------------------------------------------
def _sudah_desimal(nilai: str) -> bool:
    return "." in nilai


def _digit(nilai: str) -> tuple[int, str]:
    """Pisahkan tanda dan digit. '-1112' -> (-1, '1112')."""
    nilai = str(nilai).strip()
    tanda = -1 if nilai.startswith("-") else 1
    return tanda, nilai.lstrip("+-")


def _kandidat(nilai: str, maks_desimal: int, batas: tuple[float, float],
              izinkan_nol_depan: bool = True) -> list[float]:
    """Semua kemungkinan penempatan titik desimal yang valid."""
    tanda, d = _digit(nilai)
    hasil = []
    # titik desimal setelah digit ke-k (k = 1..len)
    for k in range(1, len(d) + 1):
        desimal = len(d) - k
        if desimal <= maks_desimal:
            hasil.append(tanda * float(d[:k] + ("." + d[k:] if d[k:] else "")))
    # angka nol di depan yang hilang (mis. -0.8826 -> '-8826')
    if izinkan_nol_depan:
        for nol in range(0, 3):
            desimal = len(d) + nol
            if desimal <= maks_desimal:
                hasil.append(tanda * float("0." + "0" * nol + d))
    lo, hi = batas
    return sorted({round(v, 6) for v in hasil if lo <= v <= hi})


def _parse_place(place: str):
    """'52 km ESE of Palu, Indonesia' -> (52.0, 112.5, 'Palu')."""
    m = re.match(r"\s*([\d.]+)\s*km\s+([NSEW]{1,3})\s+of\s+([^,]+)", str(place))
    if not m:
        return None
    return float(m.group(1)), ARAH[m.group(2)], m.group(3).strip()


def _posisi_kota_dari_gempa(lat, lon, jarak, azimut):
    """Posisi kota acuan jika gempa berada di (lat, lon)."""
    dlat = jarak * math.cos(math.radians(azimut)) / 111.2
    dlon = jarak * math.sin(math.radians(azimut)) / (111.2 * math.cos(math.radians(lat)))
    return lat - dlat, lon - dlon


def jarak_km(lat1, lon1, lat2, lon2):
    """Jarak haversine (km)."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------
# Perbaikan per kolom
# ----------------------------------------------------------------------------
def perbaiki_longitude(nilai: str) -> float:
    if _sudah_desimal(nilai):
        return float(nilai)
    tanda, d = _digit(nilai)
    return tanda * float(d[:3] + "." + d[3:]) if len(d) > 3 else tanda * float(d)


def perbaiki_magnitudo(nilai: str) -> float:
    if _sudah_desimal(nilai):
        return float(nilai)
    _, d = _digit(nilai)
    return float(d[0] + "." + d[1:]) if len(d) > 1 else float(d)


def perbaiki_latitude(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Rekonstruksi latitude dengan validasi kolom 'place'."""
    lat_akhir, catatan = [], []
    kandidat_semua = []
    for v in df["latitude"].astype(str):
        if _sudah_desimal(v):
            kandidat_semua.append([float(v)])
        else:
            kandidat_semua.append(_kandidat(v, maks_desimal=4, batas=BATAS_LAT))

    # Estimasi posisi kota acuan yang belum dikenal (mis. 'Modisi')
    kota = dict(KOTA_ACUAN)
    estimasi = {}
    for (kand, lon, place) in zip(kandidat_semua, df["lon_bersih"], df["place"]):
        info = _parse_place(place)
        if info and len(kand) == 1 and info[2] not in kota:
            estimasi.setdefault(info[2], []).append(
                _posisi_kota_dari_gempa(kand[0], lon, info[0], info[1]))
    for nama, titik in estimasi.items():
        arr = np.array(titik)
        kota[nama] = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))

    for kand, lon, place, asli in zip(kandidat_semua, df["lon_bersih"], df["place"],
                                     df["latitude"]):
        info = _parse_place(place)
        if len(kand) == 1:
            lat_akhir.append(kand[0]); catatan.append("")
            continue
        if not kand:
            lat_akhir.append(np.nan); catatan.append(f"latitude '{asli}' tidak valid")
            continue
        if info and info[2] in kota:
            klat, klon = kota[info[2]]
            # pilih kandidat yang jaraknya ke kota paling cocok dgn teks 'place'
            skor = [abs(jarak_km(c, lon, klat, klon) - info[0]) for c in kand]
            pilih = kand[int(np.argmin(skor))]
            lat_akhir.append(pilih)
            catatan.append(f"latitude ambigu {kand} -> {pilih} (validasi '{place}')")
        else:
            # tanpa info jarak: pilih yang paling dekat ke pusat wilayah studi
            pusat = -0.9
            pilih = min(kand, key=lambda c: abs(c - pusat))
            lat_akhir.append(pilih)
            catatan.append(f"latitude ambigu {kand} -> {pilih} (tanpa jarak, perkiraan)")
    return pd.Series(lat_akhir, index=df.index), pd.Series(catatan, index=df.index)


KEDALAMAN_TETAP = {"10", "35", "33"}   # kedalaman default (fixed) USGS


def perbaiki_kedalaman(df: pd.DataFrame, k: int = 5, lam: float = 1.0,
                       iterasi: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Rekonstruksi kedalaman (km).

    USGS menulis kedalaman dengan <= 3 desimal, lalu angka 0 di belakang
    dihapus (18.910 -> 18.91). Akibatnya '1891' bisa 1.891 / 18.91 / 189.1.
    Skor tiap kandidat:
        |ln(kandidat) - ln(median kedalaman k tetangga terdekat)|
        + lam * (jumlah angka 0 yang harus dianggap terhapus)
    sehingga bentuk 3 desimal lebih disukai, kecuali bertentangan kuat dengan
    pola kedalaman di sekitarnya (mis. zona subduksi dalam Teluk Tomini).
    """
    nilai = df["depth"].astype(str).tolist()
    n = len(nilai)
    kandidat, penalti = [], []
    for v in nilai:
        if _sudah_desimal(v) or v in KEDALAMAN_TETAP:
            kandidat.append([float(v)]); penalti.append([0.0]); continue
        _, d = _digit(v)
        ks, ps = [], []
        for pos in range(1, len(d) + 1):
            desimal = len(d) - pos
            if desimal > 3:
                continue
            c = float(d[:pos] + ("." + d[pos:] if d[pos:] else ""))
            if 0.5 <= c <= 700:
                ks.append(c); ps.append(lam * (3 - desimal))
        kandidat.append(ks); penalti.append(ps)

    # awal: kandidat dengan penalti terkecil (desimal terbanyak)
    hasil = np.array([ks[int(np.argmin(ps))] for ks, ps in zip(kandidat, penalti)])
    bawaan = hasil.copy()
    lat, lon = df["lat_bersih"].to_numpy(), df["lon_bersih"].to_numpy()
    jarak = np.array([[jarak_km(lat[i], lon[i], lat[j], lon[j]) for j in range(n)]
                      for i in range(n)])
    np.fill_diagonal(jarak, np.inf)
    tetangga = np.argsort(jarak, axis=1)[:, :k]

    catatan = [""] * n
    for _ in range(iterasi):
        baru = hasil.copy()
        for i in range(n):
            if len(kandidat[i]) < 2:
                continue
            acuan = float(np.median(hasil[tetangga[i]]))
            skor = [abs(math.log(c) - math.log(acuan)) + p
                    for c, p in zip(kandidat[i], penalti[i])]
            baru[i] = kandidat[i][int(np.argmin(skor))]
            catatan[i] = (f"depth '{nilai[i]}' -> {baru[i]} km bukan {bawaan[i]} "
                          f"(median {k} gempa terdekat = {acuan:.1f} km)"
                          if baru[i] != bawaan[i] else "")
        if np.allclose(baru, hasil):
            break
        hasil = baru
    return pd.Series(hasil, index=df.index), pd.Series(catatan, index=df.index)


def _aturan_pendukung(nilai, kolom):
    """Perbaikan kolom pendukung berdasarkan rentang nilai wajar USGS."""
    if pd.isna(nilai):
        return np.nan
    s = str(nilai)
    if _sudah_desimal(s):
        return float(s)
    _, d = _digit(s)
    x = float(d)
    n = len(d)
    if kolom == "rms":                 # 0.1 - 1.5 detik
        return float(d[0] + "." + d[1:]) if d[0] == "1" else float("0." + d)
    if kolom == "dmin":                # derajat, d.ddd
        return float(d[0] + "." + d[1:]) if n > 1 else x
    if kolom == "horizontalError":     # km, 2 desimal (2 - 16 km)
        return x / 100 if n >= 3 else (x / 10 if n == 2 else x)
    if kolom == "depthError":          # km, 3 desimal
        return float(d[:2] + "." + d[2:]) if n == 5 else float(d[0] + "." + d[1:])
    if kolom == "magError":            # 0.04 - 0.27
        if n == 3:
            return x / 1000
        if n == 2:
            return x / 100 if d[0] in "12" else x / 1000
        return x / 10 if d in "12" else x / 100
    return x


# ----------------------------------------------------------------------------
# Pipeline utama
# ----------------------------------------------------------------------------
def _deteksi_pemisah(path_atau_buffer) -> str:
    """Pemisah kolom: ';' (format Excel Indonesia) atau ',' (format USGS asli)."""
    if hasattr(path_atau_buffer, "read"):
        pos = path_atau_buffer.tell()
        baris = path_atau_buffer.readline()
        path_atau_buffer.seek(pos)
        if isinstance(baris, bytes):
            baris = baris.decode("utf-8-sig", errors="ignore")
    else:
        with open(path_atau_buffer, encoding="utf-8-sig") as f:
            baris = f.readline()
    return ";" if baris.count(";") > baris.count(",") else ","


def muat_dan_bersihkan(path_atau_buffer) -> tuple[pd.DataFrame, dict]:
    """Baca CSV mentah, buang baris kosong, perbaiki desimal, tambah fitur."""
    mentah = pd.read_csv(path_atau_buffer, dtype=str,
                         sep=_deteksi_pemisah(path_atau_buffer), encoding="utf-8-sig")
    n_awal = len(mentah)
    df = mentah.dropna(how="all").copy()
    df = df.dropna(subset=["latitude", "longitude", "depth", "mag"])
    n_kosong = n_awal - len(df)
    n_duplikat = int(df.duplicated(subset=["id"]).sum()) if "id" in df else 0
    if "id" in df:
        df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)

    df["lon_bersih"] = df["longitude"].astype(str).map(perbaiki_longitude)
    df["lat_bersih"], cat_lat = perbaiki_latitude(df)
    df["depth_bersih"], cat_depth = perbaiki_kedalaman(df)
    df["mag_bersih"] = df["mag"].astype(str).map(perbaiki_magnitudo)

    hasil = pd.DataFrame({
        "id": df.get("id"),
        "waktu": pd.to_datetime(df["time"], format="mixed", errors="coerce"),
        "latitude": df["lat_bersih"].round(4),
        "longitude": df["lon_bersih"].round(4),
        "kedalaman_km": df["depth_bersih"].round(3),
        "magnitudo": df["mag_bersih"].round(1),
        "tipe_magnitudo": df.get("magType"),
        "lokasi": df.get("place"),
    })
    for kol in ["nst", "gap", "dmin", "rms", "horizontalError", "depthError",
                "magError", "magNst"]:
        if kol in df:
            hasil[kol] = df[kol].map(lambda v, k=kol: _aturan_pendukung(v, k)
                                     if k not in ("nst", "gap", "magNst")
                                     else (float(v) if pd.notna(v) else np.nan))

    # Fitur turunan
    hasil["tahun"] = hasil["waktu"].dt.year
    hasil["bulan"] = hasil["waktu"].dt.month
    hasil["kelas_kedalaman"] = pd.cut(hasil["kedalaman_km"], [-1, 70, 300, 800],
                                      labels=["Dangkal (<70 km)",
                                              "Menengah (70-300 km)",
                                              "Dalam (>300 km)"])
    hasil["kelas_magnitudo"] = pd.cut(hasil["magnitudo"], [0, 4.9, 5.9, 6.9, 10],
                                      labels=["Ringan (4.0-4.9)", "Sedang (5.0-5.9)",
                                              "Kuat (6.0-6.9)", "Mayor (>=7.0)"])
    hasil["wilayah_acuan"] = hasil["lokasi"].astype(str).str.extract(
        r"of ([^,]+),")[0].fillna(hasil["lokasi"].astype(str).str.split(",").str[0])
    hasil["catatan_perbaikan"] = (cat_lat + " | " + cat_depth).str.strip(" |")

    # Validasi akhir
    valid = hasil["latitude"].between(*BATAS_LAT) & hasil["longitude"].between(*BATAS_LON)
    n_invalid = int((~valid).sum())
    hasil = hasil[valid].reset_index(drop=True)

    laporan = {
        "baris_file": n_awal,
        "baris_kosong_dibuang": n_kosong,
        "duplikat_dibuang": n_duplikat,
        "koordinat_tidak_valid": n_invalid,
        "baris_bersih": len(hasil),
        "latitude_ambigu_diselesaikan": int((cat_lat != "").sum()),
        "kedalaman_ambigu_diselesaikan": int((cat_depth != "").sum()),
    }
    return hasil, laporan


if __name__ == "__main__":
    import sys
    data, lap = muat_dan_bersihkan(sys.argv[1] if len(sys.argv) > 1 else "data/data_mentah.csv")
    print(lap)
    print(data.head(10).to_string())
