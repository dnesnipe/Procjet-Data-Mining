"""Modul Visualisasi - grafik untuk Sistem Data Mining Gempa."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Palet kategorikal (urutan tetap, aman untuk buta warna bila disertai bentuk penanda)
WARNA = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
         "#6250d6", "#e34948", "#52514e", "#104281"]
PENANDA = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})


def grafik_evaluasi(ev, k_pilih, k_siku, path=None):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, kol, judul, catatan in [
        (ax[0], "SSE", "Elbow Method (SSE)", f"siku di K={k_siku}"),
        (ax[1], "Silhouette", "Silhouette Score (tinggi = baik)", f"terbaik K={k_pilih}"),
        (ax[2], "Davies_Bouldin", "Davies-Bouldin Index (rendah = baik)", ""),
    ]:
        a.plot(ev["K"], ev[kol], color="#2a78d6", lw=2, marker="o", ms=6)
        k_tanda = k_siku if kol == "SSE" else k_pilih
        y = ev.loc[ev["K"] == k_tanda, kol].iloc[0]
        a.plot([k_tanda], [y], marker="o", ms=11, mfc="none", mec="#e34948", mew=2)
        if catatan:
            a.annotate(catatan, (k_tanda, y), textcoords="offset points",
                       xytext=(8, -16), color="#0b0b0b", fontsize=9)
        a.set_title(judul); a.set_xlabel("Jumlah cluster (K)")
        a.set_xticks(ev["K"])
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def grafik_peta(hasil, path=None):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for i, c in enumerate(sorted(hasil["cluster"].unique())):
        s = hasil[hasil["cluster"] == c]
        ax.scatter(s["longitude"], s["latitude"], s=30 + (s["magnitudo"] - 3.8) ** 2 * 30,
                   c=WARNA[i % len(WARNA)], marker=PENANDA[i % len(PENANDA)],
                   edgecolors="white", linewidths=0.8, alpha=0.9,
                   label=f"Cluster {c} ({len(s)})")
    kota = {"Palu": (-0.895, 119.859), "Poso": (-1.396, 120.753),
            "Luwuk": (-0.952, 122.788), "Masamba": (-2.553, 120.327),
            "Mamuju": (-2.675, 118.889)}
    for n, (la, lo) in kota.items():
        ax.plot(lo, la, marker="o", mfc="none", mec="#0b0b0b", ms=8, mew=1.6)
        ax.annotate(n, (lo, la), xytext=(5, -11), textcoords="offset points",
                    fontsize=8.5, color="#0b0b0b")
    ax.plot([], [], marker="o", mfc="none", mec="#0b0b0b", ls="", label="Kota acuan")
    ax.set_xlabel("Bujur (°BT)"); ax.set_ylabel("Lintang (°)")
    ax.set_title("Sebaran Episentrum Gempa per Cluster (ukuran = magnitudo)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def grafik_kedalaman_mag(hasil, path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, c in enumerate(sorted(hasil["cluster"].unique())):
        s = hasil[hasil["cluster"] == c]
        ax.scatter(s["kedalaman_km"], s["magnitudo"], s=36, c=WARNA[i % len(WARNA)],
                   marker=PENANDA[i % len(PENANDA)], edgecolors="white",
                   linewidths=0.8, label=f"Cluster {c}")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Kedalaman (km, skala log)"); ax.set_ylabel("Magnitudo")
    ax.set_title("Kedalaman vs Magnitudo per Cluster")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=9)
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig


def grafik_tahunan(hasil, path=None):
    th = hasil.groupby("tahun").size()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(th.index.astype(int).astype(str), th.values, color="#2a78d6", width=0.6)
    for x, y in zip(th.index.astype(int).astype(str), th.values):
        ax.annotate(str(y), (x, y), ha="center", va="bottom", fontsize=9,
                    xytext=(0, 2), textcoords="offset points")
    ax.set_title("Jumlah Gempa per Tahun"); ax.set_ylabel("Jumlah kejadian")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    if path:
        fig.savefig(path); plt.close(fig)
    return fig
