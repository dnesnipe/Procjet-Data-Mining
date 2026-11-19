<?php
declare(strict_types=1);

require_once __DIR__ . '/src/Preprocessing.php';
require_once __DIR__ . '/src/Clustering.php';
require_once __DIR__ . '/src/Klasifikasi.php';

use SistemGempa\Clustering;
use SistemGempa\Klasifikasi;
use SistemGempa\Preprocessing;

$cfg = require __DIR__ . '/config.php';

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
function e(mixed $v): string
{
    return htmlspecialchars((string) $v, ENT_QUOTES, 'UTF-8');
}

function fmt(float|int $v, int $desimal = 2): string
{
    return number_format((float) $v, $desimal, ',', '.');
}

function kartu(string $label, string $nilai): string
{
    return '<div class="kartu"><div class="lbl">' . e($label) . '</div><div class="nilai">' . e($nilai) . '</div></div>';
}

/** @param list<string> $head @param list<list<string>> $rows (sudah di-escape) */
function tabel(array $head, array $rows, string $kelas = ''): string
{
    $h = '<div class="tabel-wrap"><table class="' . $kelas . '"><thead><tr>';
    foreach ($head as $c) {
        $h .= '<th>' . e($c) . '</th>';
    }
    $h .= '</tr></thead><tbody>';
    foreach ($rows as $r) {
        $h .= '<tr>';
        foreach ($r as $cell) {
            $h .= '<td>' . $cell . '</td>';
        }
        $h .= '</tr>';
    }
    return $h . '</tbody></table></div>';
}

/** Dataset scatter untuk Chart.js, satu dataset per cluster. */
function datasetScatter(array $hasil, string $kx, string $ky, array $warna): array
{
    $out = [];
    foreach ($hasil as $r) {
        $c = $r['cluster'];
        if (!isset($out[$c])) {
            $out[$c] = [
                'label'           => 'Cluster ' . $c,
                'data'            => [],
                'backgroundColor' => $warna[($c - 1) % count($warna)],
                'pointRadius'     => 4,
            ];
        }
        $out[$c]['data'][] = ['x' => (float) $r[$kx], 'y' => (float) $r[$ky]];
    }
    ksort($out);
    return array_values($out);
}

function hiddenFilter(): string
{
    global $norm, $wilayahPilih, $kInput, $magMinStr, $dari, $sampai;
    return '<input type="hidden" name="norm" value="' . e($norm) . '">'
        . '<input type="hidden" name="wilayah" value="' . e($wilayahPilih) . '">'
        . '<input type="hidden" name="k" value="' . (int) $kInput . '">'
        . '<input type="hidden" name="magmin" value="' . e($magMinStr) . '">'
        . '<input type="hidden" name="dari" value="' . e($dari) . '">'
        . '<input type="hidden" name="sampai" value="' . e($sampai) . '">';
}

/** Saring data berdasarkan magnitudo minimum dan rentang tanggal (YYYY-MM-DD). */
function filterData(array $data, float $magMin, string $dari, string $sampai): array
{
    $tDari   = $dari !== '' ? strtotime($dari . ' 00:00:00') : false;
    $tSampai = $sampai !== '' ? strtotime($sampai . ' 23:59:59') : false;
    return array_values(array_filter($data, static function (array $r) use ($magMin, $tDari, $tSampai): bool {
        if ($r['magnitudo'] < $magMin) {
            return false;
        }
        if ($tDari === false && $tSampai === false) {
            return true;
        }
        $ts = strtotime((string) $r['waktu']);
        if ($ts === false) {
            return false;
        }
        if ($tDari !== false && $ts < $tDari) {
            return false;
        }
        if ($tSampai !== false && $ts > $tSampai) {
            return false;
        }
        return true;
    }));
}

function badge(string $tingkat): string
{
    $kls = ['Tinggi' => 'tinggi', 'Sedang' => 'sedang', 'Rendah' => 'rendah'][$tingkat] ?? '';
    return '<span class="badge ' . $kls . '">' . e($tingkat) . '</span>';
}

$warna = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#7c3aed',
          '#0891b2', '#db2777', '#65a30d', '#ea580c', '#475569'];

// ---------------------------------------------------------------------------
// Aksi POST: upload / reset dataset
// ---------------------------------------------------------------------------
$uploadPath = $cfg['data_upload'];
$error = '';

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $aksi = $_POST['aksi'] ?? '';

    if ($aksi === 'reset') {
        if (is_file($uploadPath)) {
            unlink($uploadPath);
        }
        header('Location: index.php?menu=1&info=reset');
        exit;
    }

    if ($aksi === 'upload') {
        $f = $_FILES['csv'] ?? null;
        if ($f === null || $f['error'] !== UPLOAD_ERR_OK) {
            $error = 'Upload gagal. Pastikan file sudah dipilih.';
        } elseif (strtolower(pathinfo((string) $f['name'], PATHINFO_EXTENSION)) !== 'csv') {
            $error = 'File harus berekstensi .csv';
        } elseif ($f['size'] > $cfg['upload_maks_byte']) {
            $error = 'Ukuran file melebihi 5 MB.';
        } elseif (!is_dir(dirname($uploadPath)) && !mkdir(dirname($uploadPath), 0775, true)) {
            $error = 'Folder data tidak dapat dibuat.';
        } elseif (!move_uploaded_file($f['tmp_name'], $uploadPath)) {
            $error = 'File tidak dapat disimpan ke server.';
        } else {
            header('Location: index.php?menu=1&info=upload');
            exit;
        }
    }
}

// ---------------------------------------------------------------------------
// Parameter dari URL
// ---------------------------------------------------------------------------
$info    = $_GET['info'] ?? '';
$menu    = (int) ($_GET['menu'] ?? 1);
if ($menu < 1 || $menu > 7) {
    $menu = 1;
}
$norm           = ($_GET['norm'] ?? 'zscore') === 'minmax' ? 'minmax' : 'zscore';
$wilayahPilih  = trim((string) ($_GET['wilayah'] ?? 'semua'));
$tanpaGorontalo = false; // pemilihan wilayah dilakukan dari daftar nama di data
$kInput         = max(0, min(10, (int) ($_GET['k'] ?? 0)));
$fitur          = $cfg['fitur'];
$magMinStr = trim((string) ($_GET['magmin'] ?? '0'));
$magMin    = is_numeric($magMinStr) ? max(0.0, min(9.0, (float) $magMinStr)) : 0.0;
$magMinStr = (string) $magMin;
$dari      = preg_match('/^\d{4}-\d{2}-\d{2}$/', (string) ($_GET['dari'] ?? '')) ? (string) $_GET['dari'] : '';
$sampai    = preg_match('/^\d{4}-\d{2}-\d{2}$/', (string) ($_GET['sampai'] ?? '')) ? (string) $_GET['sampai'] : '';

$sumber     = is_file($uploadPath) ? $uploadPath : $cfg['data_default'];
$namaSumber = basename($sumber);

// ---------------------------------------------------------------------------
// Pipeline: preprocessing -> evaluasi K -> K-Means
// ---------------------------------------------------------------------------
$data = [];
$lap = [];
$ev = [];
$kOpt = 0;
$kSiku = 0;
$k = 0;
$analisis = null;
$daftarWilayah = [];
$judulWilayah = 'Semua wilayah';

try {
    [$data, $lap] = Preprocessing::muatDanBersihkan($sumber, $tanpaGorontalo);
    $hitungWil = array_count_values(array_column($data, 'wilayah_acuan'));
    arsort($hitungWil);
    $daftarWilayah = $hitungWil;
    $data = filterData($data, $magMin, $dari, $sampai);
    if ($wilayahPilih !== 'semua' && isset($daftarWilayah[$wilayahPilih])) {
        $data = array_values(array_filter($data, static fn ($r) => $r['wilayah_acuan'] === $wilayahPilih));
    } else {
        $wilayahPilih = 'semua';
    }
    $judulWilayah = $wilayahPilih === 'semua' ? 'Semua wilayah' : $wilayahPilih;
    if (count($data) < 10) {
        throw new RuntimeException('Data bersih kurang dari 10 baris; analisis tidak dapat dijalankan.');
    }
    [$X] = Clustering::skala($data, $fitur, $norm);
    $ev    = Clustering::evaluasiK($X, 2, $cfg['k_maks']);
    $kOpt  = Clustering::pilihK($ev);
    $kSiku = Clustering::titikSiku($ev);
    $k     = $kInput >= 2 ? min($kInput, count($data)) : $kOpt;
    $analisis = Clustering::jalankan($data, $fitur, $norm, $k);
} catch (Throwable $ex) {
    $error = $error ?: $ex->getMessage();
}

// ---------------------------------------------------------------------------
// Unduh CSV (harus sebelum output HTML)
// ---------------------------------------------------------------------------
if ($analisis !== null && isset($_GET['unduh'])) {
    $jenis = $_GET['unduh'] === 'profil' ? 'profil' : 'hasil';
    $rows  = $jenis === 'profil' ? $analisis['profil'] : $analisis['hasil'];
    header('Content-Type: text/csv; charset=utf-8');
    header('Content-Disposition: attachment; filename="' . $jenis . '_cluster.csv"');
    $out = fopen('php://output', 'w');
    fwrite($out, "\xEF\xBB\xBF");
    if ($rows) {
        fputcsv($out, array_keys($rows[0]));
        foreach ($rows as $r) {
            fputcsv($out, $r);
        }
    }
    fclose($out);
    exit;
}

// Grafik Chart.js dikumpulkan, dicetak di akhir halaman
$charts = [];

$menuList = [
    1 => 'Dataset',
    2 => 'Preprocessing',
    3 => 'Eksplorasi Data',
    4 => 'Penentuan K',
    5 => 'Clustering',
    6 => 'Prediksi Cluster',
    7 => 'Klasifikasi (k-NN)',
];
$qsFilter = http_build_query([
    'norm'    => $norm,
    'wilayah' => $wilayahPilih,
    'k'       => $kInput,
    'magmin'  => $magMinStr,
    'dari'    => $dari,
    'sampai'  => $sampai,
]);
?>
<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Data Mining Gempa Sulawesi</title>
<style>
  :root { --bg:#f3f5f9; --card:#fff; --txt:#1f2937; --mut:#6b7280; --line:#e5e7eb;
          --merah:#c8102e; --navy:#1e2330; --acc:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; background:var(--bg); color:var(--txt); }
  .topbar { background:var(--navy); color:#fff; padding:16px 28px; }
  .topbar h1 { margin:0; font-size:20px; }
  .topbar p { margin:4px 0 0; font-size:13px; color:#9ca3af; }
  .wrap { max-width:1280px; margin:0 auto; padding:18px 24px 40px; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:14px; }
  .tab { padding:8px 14px; border-radius:8px; background:#fff; border:1px solid var(--line); color:var(--txt); text-decoration:none; font-size:14px; }
  .tab.aktif { background:var(--navy); color:#fff; border-color:var(--navy); }
  .filter { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px;
            display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; margin-bottom:16px; }
  .filter label { font-size:12px; color:var(--mut); display:flex; flex-direction:column; gap:4px; }
  .filter input, .filter select { padding:8px 10px; border:1px solid var(--line); border-radius:8px; font-size:14px; min-width:130px; background:#fff; }
  .btn-merah { background:var(--merah); color:#fff; border:0; border-radius:8px; padding:9px 18px; font-size:14px; cursor:pointer; }
  .reset { font-size:13px; color:var(--txt); padding-bottom:9px; }
  .kpi { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:16px; }
  .kpi .card { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }
  .kpi .nilai { font-size:26px; font-weight:700; }
  .kpi .lbl { font-size:13px; color:var(--mut); margin-top:2px; }
  .kpi .sub { font-size:12px; color:var(--mut); margin-top:4px; }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px; }
  .panel h3 { margin:0 0 10px; font-size:15px; }
  .panel .judul-kecil { font-size:12px; color:var(--mut); font-weight:400; }
  #peta { height:420px; border-radius:10px; }
  h2 { margin:0 0 6px; font-size:22px; }
  h3 { font-size:16px; margin:24px 0 10px; }
  .sub { color:var(--mut); font-size:14px; margin-bottom:18px; }
  .kartu-wrap { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:18px; }
  .kartu { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .kartu .lbl { font-size:12px; color:var(--mut); }
  .kartu .nilai { font-size:20px; font-weight:600; margin-top:4px; }
  .box { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; margin-bottom:16px; }
  .alert { padding:12px 14px; border-radius:8px; margin-bottom:16px; font-size:14px; }
  .alert.err { background:#fee2e2; color:#7f1d1d; }
  .alert.ok { background:#dcfce7; color:#14532d; }
  .alert.warn { background:#fef3c7; color:#78350f; }
  .tabel-wrap { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:12px; margin-bottom:16px; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
  th { background:#f8fafc; font-weight:600; }
  tr.best td { background:#dbeafe; }
  .chart { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px; margin-bottom:16px; }
  .grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }
  .badge { padding:3px 9px; border-radius:999px; font-size:12px; font-weight:600; }
  .badge.tinggi { background:#fee2e2; color:#991b1b; }
  .badge.sedang { background:#fef3c7; color:#92400e; }
  .badge.rendah { background:#dcfce7; color:#166534; }
  ul.langkah { padding-left:18px; line-height:1.7; font-size:14px; }
  .muted { color:var(--mut); font-size:13px; }
  .legenda span { margin-right:12px; font-size:13px; }
  .btn { padding:8px 14px; border:0; border-radius:8px; background:var(--acc); color:#fff; font-size:14px; cursor:pointer; text-decoration:none; display:inline-block; }
  .btn.ghost { background:#e5e7eb; color:var(--txt); }
  @media (max-width: 800px) {
    .wrap { padding:14px; }
    .topbar { padding:14px 16px; }
    .kpi .nilai { font-size:22px; }
  }
</style>
</head>
<body>
<header class="topbar">
  <h1>Sistem Informasi Gempa Sulawesi</h1>
  <p>Data USGS (2020 – 2026) · analisis statistik &amp; klasterisasi K-Means</p>
</header>
<div class="wrap">
<nav class="tabs"><?php foreach ($menuList as $no => $nama): ?><a class="tab<?= $no === $menu ? ' aktif' : '' ?>" href="?menu=<?= $no ?>&amp;<?= e($qsFilter) ?>"><?= $no ?>. <?= e($nama) ?></a><?php endforeach; ?></nav>
<?php if ($info === 'upload'): ?>
  <div class="alert ok">Dataset berhasil diunggah dan dipakai untuk analisis.</div>
<?php elseif ($info === 'reset'): ?>
  <div class="alert ok">Kembali menggunakan dataset default.</div>
<?php endif; ?>

<?php if ($error !== ''): ?>
  <div class="alert err"><?= e($error) ?></div>
<?php endif; ?>

<form class="filter" method="get">
  <input type="hidden" name="menu" value="<?= $menu ?>">
  <input type="hidden" name="norm" value="<?= e($norm) ?>">
  <label>Wilayah
    <select name="wilayah">
      <option value="semua" <?= $wilayahPilih === 'semua' ? 'selected' : '' ?>>Semua wilayah</option>
      <?php foreach ($daftarWilayah as $namaWil => $jumlahWil): ?>
      <option value="<?= e($namaWil) ?>" <?= $wilayahPilih === (string) $namaWil ? 'selected' : '' ?>><?= e($namaWil) ?> (<?= (int) $jumlahWil ?>)</option>
      <?php endforeach; ?>
    </select>
  </label>
  <label>Magnitudo minimum
    <input type="number" name="magmin" step="0.1" min="0" max="9" value="<?= e($magMinStr) ?>">
  </label>
  <label>Dari tanggal
    <input type="date" name="dari" value="<?= e($dari) ?>">
  </label>
  <label>Sampai tanggal
    <input type="date" name="sampai" value="<?= e($sampai) ?>">
  </label>
  <label>Jumlah klaster (K)
    <input type="number" name="k" min="0" max="10" value="<?= $kInput ?>">
  </label>
  <button class="btn-merah" type="submit">Terapkan</button>
  <a class="reset" href="?menu=<?= $menu ?>">Reset</a>
</form>
<?php if ($data): ?>
<?php
  $magArr  = array_column($data, 'magnitudo');
  $iMax    = array_keys($magArr, max($magArr))[0];
  $terkuat = $data[$iMax];
  $tsKuat  = strtotime((string) $terkuat['waktu']);
  $titik   = [];
  foreach (($analisis['hasil'] ?? $data) as $r) {
      $c = (int) ($r['cluster'] ?? 0);
      $titik[] = [
          'lat'   => $r['latitude'],
          'lon'   => $r['longitude'],
          'mag'   => $r['magnitudo'],
          'color' => $c > 0 ? $warna[($c - 1) % count($warna)] : '#6b7280',
          'label' => 'M' . number_format($r['magnitudo'], 1, '.', '') . ' · ' . $r['lokasi'] . ($c > 0 ? ' · Cluster ' . $c : ''),
      ];
  }
?>
<div class="kpi">
  <div class="card"><div class="nilai"><?= fmt(count($data), 0) ?></div><div class="lbl">Total kejadian</div><div class="sub">Wilayah: <?= e($judulWilayah) ?></div></div>
  <div class="card"><div class="nilai"><?= fmt(array_sum($magArr) / count($magArr), 2) ?></div><div class="lbl">Rata-rata magnitudo</div></div>
  <div class="card"><div class="nilai"><?= fmt($terkuat['magnitudo'], 1) ?></div><div class="lbl">Terkuat</div><div class="sub"><?= $tsKuat ? e(date('Y-m-d', $tsKuat)) : '-' ?></div></div>
  <div class="card"><div class="nilai"><?= fmt(array_sum(array_column($data, 'kedalaman_km')) / count($data), 1) ?> km</div><div class="lbl">Rata-rata kedalaman</div></div>
</div>
<div class="panel">
  <h3>Peta Episenter · <?= e($judulWilayah) ?> <span class="judul-kecil">(warna = klaster, ukuran = magnitudo)</span></h3>
  <div id="peta"></div>
</div>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<script>
(function () {
  var pts = <?= json_encode($titik, JSON_HEX_TAG | JSON_HEX_AMP | JSON_THROW_ON_ERROR) ?>;
  var peta = L.map('peta').setView([-1.5, 121.5], 7);
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 18, attribution: '&copy; OpenStreetMap'
  }).addTo(peta);
  pts.forEach(function (p) {
    L.circleMarker([p.lat, p.lon], {
      radius: Math.max(3, (p.mag - 3.5) * 3 + 3), color: p.color, fillColor: p.color,
      fillOpacity: 0.6, weight: 1
    }).bindTooltip(p.label).addTo(peta);
  });
})();
</script>
<?php endif; ?>

<?php
// =========================================================================
// MENU 1 - DATASET
// =========================================================================
if ($menu === 1):
?>
  <h2>1. Dataset Gempa Bumi</h2>
  <p class="sub">Katalog USGS (earthquake.usgs.gov). Unggah CSV dengan format yang sama untuk dianalisis.</p>

  <?php if ($lap): ?>
  <div class="kartu-wrap">
    <?= kartu('Baris pada file', fmt($lap['baris_pada_file'], 0)) ?>
    <?= kartu('Baris bersih', fmt($lap['baris_bersih'], 0)) ?>
    <?= kartu('Periode', (string) $lap['periode']) ?>
    <?= kartu('Magnitudo rata-rata', $data ? 'M' . fmt(array_sum(array_column($data, 'magnitudo')) / count($data)) : '-') ?>
  </div>
  <?php endif; ?>

  <div class="box">
    <h3 style="margin-top:0">Unggah dataset baru</h3>
    <form method="post" enctype="multipart/form-data" style="display:flex; gap:10px; flex-wrap:wrap; align-items:center">
      <input type="hidden" name="aksi" value="upload">
      <input type="file" name="csv" accept=".csv" required>
      <button type="submit">⬆️ Unggah & gunakan</button>
    </form>
    <?php if (is_file($uploadPath)): ?>
    <form method="post" style="margin-top:10px">
      <input type="hidden" name="aksi" value="reset">
      <button type="submit" class="btn ghost">Kembali ke dataset default</button>
    </form>
    <?php endif; ?>
  </div>

  <?php if ($data): ?>
  <h3>Data bersih (100 baris pertama)</h3>
  <?php
  $baris = [];
  foreach (array_slice($data, 0, 100) as $r) {
      $baris[] = [
          e($r['waktu']), e(fmt($r['latitude'], 4)), e(fmt($r['longitude'], 4)),
          e(fmt($r['kedalaman_km'], 1)), e(fmt($r['magnitudo'], 1)), e($r['lokasi']),
      ];
  }
  echo tabel(['Waktu (UTC)', 'Latitude', 'Longitude', 'Kedalaman (km)', 'Magnitudo', 'Lokasi'], $baris);
  ?>
  <?php endif; ?>

<?php
// =========================================================================
// MENU 2 - PREPROCESSING
// =========================================================================
elseif ($menu === 2):
?>
  <h2>2. Preprocessing Data</h2>
  <p class="sub">Pembersihan data sebelum clustering.</p>
  <div class="box">
    <ul class="langkah">
      <li><b>Data cleaning</b> — membuang baris kosong, duplikat berdasarkan <code>id</code>, dan (opsional) gempa dengan keterangan wilayah Gorontalo.</li>
      <li><b>Validasi koordinat</b> — latitude −4 s.d. 3, longitude 118 s.d. 126, kedalaman ≥ 0.</li>
      <li><b>Fitur turunan</b> — tahun, kelas kedalaman (dangkal &lt;70 km, menengah 70–300 km, dalam &gt;300 km), kelas magnitudo, dan wilayah acuan dari kolom <code>place</code>.</li>
    </ul>
  </div>
  <?php if ($lap): ?>
  <?= tabel(['Keterangan', 'Nilai'], array_map(
        static fn ($k, $v) => [e(str_replace('_', ' ', $k)), e(is_int($v) ? fmt($v, 0) : (string) $v)],
        array_keys($lap), array_values($lap))) ?>
  <div class="alert warn"><?= e($lap['catatan']) ?></div>
  <?php endif; ?>

<?php
// =========================================================================
// MENU 3 - EKSPLORASI
// =========================================================================
elseif ($menu === 3):
  if ($data):
    $dangkal = count(array_filter($data, static fn ($r) => $r['kedalaman_km'] < 70));
    $perTahun = array_count_values(array_column($data, 'tahun'));
    ksort($perTahun);
    $kelasDep = array_count_values(array_column($data, 'kelas_kedalaman'));
    $kelasMag = array_count_values(array_column($data, 'kelas_magnitudo'));
    $wilayah  = array_count_values(array_column($data, 'wilayah_acuan'));
    arsort($wilayah);
    $wilayah  = array_slice($wilayah, 0, 10, true);
    $terbesar = $data;
    usort($terbesar, static fn ($a, $b) => $b['magnitudo'] <=> $a['magnitudo']);
    $desk = Clustering::deskriptif($data, $fitur);
?>
  <h2>3. Eksplorasi Data (EDA)</h2>
  <div class="kartu-wrap">
    <?= kartu('Total gempa', fmt(count($data), 0)) ?>
    <?= kartu('Magnitudo maks', 'M' . fmt(max(array_column($data, 'magnitudo')), 1)) ?>
    <?= kartu('Kedalaman rata-rata', fmt(array_sum(array_column($data, 'kedalaman_km')) / count($data), 1) . ' km') ?>
    <?= kartu('Gempa dangkal (<70 km)', fmt($dangkal / count($data) * 100, 0) . '%') ?>
  </div>

  <h3>Statistik deskriptif</h3>
  <?php
  $rowsDesk = [];
  foreach ($desk as $f => $s) {
      $rowsDesk[] = [e($f), e(fmt($s['jumlah'], 0)), e(fmt($s['rata2'], 3)), e(fmt($s['std'], 3)),
          e(fmt($s['min'], 3)), e(fmt($s['p25'], 3)), e(fmt($s['p50'], 3)), e(fmt($s['p75'], 3)), e(fmt($s['maks'], 3))];
  }
  echo tabel(['Fitur', 'Jumlah', 'Rata-rata', 'Std', 'Min', '25%', '50%', '75%', 'Maks'], $rowsDesk);
  ?>

  <div class="grid2">
    <div class="chart">
      <h3 style="margin-top:0">Jumlah gempa per tahun</h3>
      <canvas id="c_tahun" height="200"></canvas>
      <?php $charts['c_tahun'] = ['type' => 'bar', 'data' => [
          'labels' => array_map('strval', array_keys($perTahun)),
          'datasets' => [['label' => 'Jumlah gempa', 'data' => array_values($perTahun), 'backgroundColor' => '#2563eb']],
      ]]; ?>
    </div>
    <div class="chart">
      <h3 style="margin-top:0">Gempa per wilayah acuan (10 teratas)</h3>
      <canvas id="c_wil" height="200"></canvas>
      <?php $charts['c_wil'] = ['type' => 'bar', 'data' => [
          'labels' => array_keys($wilayah),
          'datasets' => [['label' => 'Jumlah', 'data' => array_values($wilayah), 'backgroundColor' => '#0891b2']],
      ], 'options' => ['indexAxis' => 'y']]; ?>
    </div>
  </div>

  <div class="grid2">
    <div>
      <h3>Kelas kedalaman</h3>
      <?php echo tabel(['Kelas', 'Jumlah'], array_map(static fn ($k, $v) => [e($k), e(fmt($v, 0))],
          array_keys($kelasDep), array_values($kelasDep))); ?>
    </div>
    <div>
      <h3>Kelas magnitudo</h3>
      <?php echo tabel(['Kelas', 'Jumlah'], array_map(static fn ($k, $v) => [e($k), e(fmt($v, 0))],
          array_keys($kelasMag), array_values($kelasMag))); ?>
    </div>
  </div>

  <h3>10 gempa terbesar</h3>
  <?php
  $rowsBesar = [];
  foreach (array_slice($terbesar, 0, 10) as $r) {
      $rowsBesar[] = [e($r['waktu']), e(fmt($r['magnitudo'], 1)), e(fmt($r['kedalaman_km'], 1)), e($r['lokasi'])];
  }
  echo tabel(['Waktu (UTC)', 'Magnitudo', 'Kedalaman (km)', 'Lokasi'], $rowsBesar);
  ?>
<?php endif;

// =========================================================================
// MENU 4 - PENENTUAN K
// =========================================================================
elseif ($menu === 4):
  if ($ev):
?>
  <h2>4. Penentuan Jumlah Cluster Optimal</h2>
  <div class="box">
    <ul class="langkah">
      <li><b>Elbow (SSE)</b> — titik di mana penurunan SSE mulai melandai (titik siku).</li>
      <li><b>Silhouette Score</b> — kerapatan & keterpisahan cluster (−1 s.d. 1, makin tinggi makin baik).</li>
      <li><b>Davies-Bouldin Index</b> — makin rendah makin baik.</li>
    </ul>
  </div>
  <div class="grid2">
    <div class="chart">
      <h3 style="margin-top:0">SSE (Elbow)</h3>
      <canvas id="c_sse" height="200"></canvas>
      <?php $charts['c_sse'] = ['type' => 'line', 'data' => [
          'labels' => array_column($ev, 'K'),
          'datasets' => [['label' => 'SSE', 'data' => array_column($ev, 'SSE'), 'borderColor' => '#2563eb', 'tension' => 0.2]],
      ]]; ?>
    </div>
    <div class="chart">
      <h3 style="margin-top:0">Silhouette & Davies-Bouldin</h3>
      <canvas id="c_sil" height="200"></canvas>
      <?php $charts['c_sil'] = ['type' => 'line', 'data' => [
          'labels' => array_column($ev, 'K'),
          'datasets' => [
              ['label' => 'Silhouette', 'data' => array_column($ev, 'Silhouette'), 'borderColor' => '#16a34a', 'tension' => 0.2],
              ['label' => 'Davies-Bouldin', 'data' => array_column($ev, 'Davies_Bouldin'), 'borderColor' => '#dc2626', 'tension' => 0.2],
          ],
      ]]; ?>
    </div>
  </div>
  <?php
  $rowsEv = [];
  foreach ($ev as $r) {
      $tanda = $r['K'] === $kOpt ? ' ★ optimal' : '';
      $tanda .= $r['K'] === $kSiku ? ' (siku Elbow)' : '';
      $rowsEv[] = [e($r['K']) . e($tanda), e(fmt($r['SSE'], 4)), e(fmt($r['Silhouette'], 4)), e(fmt($r['Davies_Bouldin'], 4))];
  }
  echo tabel(['K', 'SSE', 'Silhouette', 'Davies-Bouldin'], $rowsEv);
  ?>
  <div class="alert ok">Silhouette tertinggi di <b>K = <?= $kOpt ?></b>
    (<?= fmt(max(array_column($ev, 'Silhouette')), 4) ?>); titik siku Elbow di <b>K = <?= $kSiku ?></b>.</div>
<?php endif;

// =========================================================================
// MENU 5 - CLUSTERING
// =========================================================================
elseif ($menu === 5):
  if ($analisis):
    $hasil  = $analisis['hasil'];
    $profil = $analisis['profil'];
    $met    = $analisis['metrik'];
?>
  <h2>5. Hasil K-Means Clustering (K = <?= $k ?>)</h2>
  <div class="kartu-wrap">
    <?= kartu('Jumlah cluster', (string) $k) ?>
    <?= kartu('Silhouette', fmt($met['Silhouette'], 4)) ?>
    <?= kartu('Davies-Bouldin', fmt($met['Davies_Bouldin'], 4)) ?>
    <?= kartu('SSE', fmt($met['SSE'], 2)) ?>
  </div>

  <div class="legenda">
    <?php foreach (range(1, $k) as $c): ?>
      <span style="color:<?= $warna[($c - 1) % count($warna)] ?>">●</span> Cluster <?= $c ?>
    <?php endforeach; ?>
  </div>

  <div class="grid2">
    <div class="chart">
      <h3 style="margin-top:0">Peta sebaran episentrum (bujur × lintang)</h3>
      <canvas id="c_peta" height="260"></canvas>
      <?php $charts['c_peta'] = ['type' => 'scatter', 'data' => ['datasets' => datasetScatter($hasil, 'longitude', 'latitude', $warna)],
          'options' => ['scales' => ['x' => ['title' => ['display' => true, 'text' => 'Bujur']],
                                     'y' => ['title' => ['display' => true, 'text' => 'Lintang']]]]]; ?>
    </div>
    <div class="chart">
      <h3 style="margin-top:0">Kedalaman × magnitudo</h3>
      <canvas id="c_dm" height="260"></canvas>
      <?php $charts['c_dm'] = ['type' => 'scatter', 'data' => ['datasets' => datasetScatter($hasil, 'kedalaman_km', 'magnitudo', $warna)],
          'options' => ['scales' => ['x' => ['title' => ['display' => true, 'text' => 'Kedalaman (km)']],
                                     'y' => ['title' => ['display' => true, 'text' => 'Magnitudo']]]]]; ?>
    </div>
  </div>

  <h3>Profil & tingkat kerawanan cluster</h3>
  <?php
  $rowsP = [];
  foreach ($profil as $p) {
      $rowsP[] = [e($p['cluster']), e(fmt($p['jumlah_gempa'], 0)), e(fmt($p['persen'], 2)),
          e(fmt($p['kedalaman_rata2_km'], 1)), e(fmt($p['mag_rata2'], 2)), e(fmt($p['mag_maks'], 1)),
          e($p['wilayah_dominan']), e(fmt($p['skor_kerawanan'], 4)), badge($p['tingkat_kerawanan'])];
  }
  echo tabel(['Cluster', 'Gempa', '%', 'Kedalaman rata² (km)', 'Mag rata²', 'Mag maks', 'Wilayah dominan', 'Skor', 'Kerawanan'], $rowsP);
  ?>
  <div class="box">
    <?php foreach ($profil as $p): ?>
      <p style="margin:6px 0">• <b>Cluster <?= $p['cluster'] ?></b> — <?= e($p['jumlah_gempa']) ?> gempa,
        kerawanan <b><?= e($p['tingkat_kerawanan']) ?></b>. <?= e($p['karakteristik']) ?>.</p>
    <?php endforeach; ?>
    <p class="muted">Skor kerawanan = 0,40 × frekuensi (ternormalisasi) + 0,35 × magnitudo rata-rata (ternormalisasi)
      + 0,25 × proporsi gempa dangkal (&lt;70 km). Tinggi &gt; 0,65 · Sedang 0,40–0,65 · Rendah &lt; 0,40.</p>
  </div>

  <p>
    <a class="btn" href="?menu=5&amp;unduh=hasil&amp;<?= e($qsFilter) ?>">⬇️ Unduh hasil clustering (CSV)</a>
    <a class="btn ghost" href="?menu=5&amp;unduh=profil&amp;<?= e($qsFilter) ?>">⬇️ Unduh profil cluster (CSV)</a>
  </p>

  <h3>Data hasil clustering</h3>
  <?php
  $rowsH = [];
  foreach ($hasil as $r) {
      $rowsH[] = [e($r['cluster']), e($r['waktu']), e(fmt($r['latitude'], 4)), e(fmt($r['longitude'], 4)),
          e(fmt($r['kedalaman_km'], 1)), e(fmt($r['magnitudo'], 1)), e($r['lokasi'])];
  }
  echo tabel(['Cluster', 'Waktu (UTC)', 'Latitude', 'Longitude', 'Kedalaman (km)', 'Magnitudo', 'Lokasi'], $rowsH);
  ?>
<?php endif;

// =========================================================================
// MENU 6 - PREDIKSI
// =========================================================================
elseif ($menu === 6):
  if ($analisis):
    $batas = [
        'latitude'     => ['Latitude', -4.0, 3.0, 0.0001],
        'longitude'    => ['Longitude', 118.0, 126.0, 0.0001],
        'kedalaman_km' => ['Kedalaman (km)', 0.0, 700.0, 0.1],
        'magnitudo'    => ['Magnitudo', 3.0, 9.0, 0.1],
    ];
    $nilaiForm = [];
    $pesanPred = '';
    $hasilPred = null;
    $submit = isset($_GET['prediksi']);

    foreach ($fitur as $f) {
        $rata = round(array_sum(array_column($data, $f)) / count($data), 4);
        $raw  = $_GET[$f] ?? (string) $rata;
        $nilaiForm[$f] = $raw;
    }

    if ($submit) {
        $nilai = [];
        foreach ($fitur as $f) {
            [, $lo, $hi] = $batas[$f];
            $v = $_GET[$f] ?? '';
            if (!is_numeric($v) || (float) $v < $lo || (float) $v > $hi) {
                $pesanPred = "Nilai {$batas[$f][0]} harus antara {$lo} dan {$hi}.";
                break;
            }
            $nilai[$f] = (float) $v;
        }
        if ($pesanPred === '') {
            $c = Clustering::prediksi($nilai, $fitur, $analisis['params'], $analisis['centroid']);
            $hasilPred = $analisis['profil'][$c - 1];
        }
    }
?>
  <h2>6. Prediksi Cluster Gempa Baru</h2>
  <p class="sub">Masukkan parameter gempa baru. Sistem menentukan cluster dengan centroid terdekat dari model K-Means yang sudah dilatih.</p>

  <form class="box" method="get" style="display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; align-items:end">
    <input type="hidden" name="menu" value="6">
    <?= hiddenFilter() ?>
    <?php foreach ($fitur as $f): [$label, $lo, $hi, $step] = $batas[$f]; ?>
      <label class="muted" style="display:flex; flex-direction:column; gap:4px">
        <?= e($label) ?> (<?= $lo ?> … <?= $hi ?>)
        <input type="number" step="<?= $step ?>" min="<?= $lo ?>" max="<?= $hi ?>" name="<?= e($f) ?>" value="<?= e($nilaiForm[$f]) ?>" required>
      </label>
    <?php endforeach; ?>
    <div><button type="submit" name="prediksi" value="1">🔍 Prediksi cluster</button></div>
  </form>

  <?php if ($pesanPred !== ''): ?>
    <div class="alert err"><?= e($pesanPred) ?></div>
  <?php elseif ($hasilPred !== null): ?>
    <div class="alert ok">Gempa ini masuk <b>Cluster <?= $hasilPred['cluster'] ?></b> — tingkat kerawanan <b><?= e($hasilPred['tingkat_kerawanan']) ?></b>.</div>
    <div class="box"><?= e($hasilPred['karakteristik']) ?>.</div>
    <?php
    $rowsPr = [[e($hasilPred['cluster']), e(fmt($hasilPred['jumlah_gempa'], 0)), e(fmt($hasilPred['kedalaman_rata2_km'], 1)),
        e(fmt($hasilPred['mag_rata2'], 2)), e(fmt($hasilPred['skor_kerawanan'], 4)), badge($hasilPred['tingkat_kerawanan'])]];
    echo tabel(['Cluster', 'Gempa', 'Kedalaman rata² (km)', 'Mag rata²', 'Skor', 'Kerawanan'], $rowsPr);
    ?>
  <?php endif; ?>
<?php endif;
endif;

// =========================================================================
// MENU 7 - KLASIFIKASI k-NN (target: kelas kedalaman)
// =========================================================================
if ($menu === 7):
  $kk = max(1, min(25, (int) ($_GET['kk'] ?? 5)));
  $klas = null; $errKl = '';
  if ($data) {
      try {
          $klas = Klasifikasi::knn($data, ['latitude', 'longitude', 'magnitudo'], 'kelas_kedalaman', $kk, 0.3, $cfg['seed']);
      } catch (Throwable $ex) {
          $errKl = $ex->getMessage();
      }
  }
?>
  <h2>7. Klasifikasi k-NN: Kelas Kedalaman Gempa</h2>
  <div class="box">
    <p class="sub" style="margin:0">Model memprediksi kelas kedalaman (Dangkal / Menengah / Dalam) dari
      latitude, longitude, dan magnitudo. Kedalaman tidak dipakai sebagai fitur karena menjadi target.
      Data dibagi 70% latih dan 30% uji secara acak dengan seed tetap.</p>
  </div>
  <form class="filter" method="get">
    <input type="hidden" name="menu" value="7">
    <?= hiddenFilter() ?>
    <label>Jumlah tetangga k (1–25)
      <input type="number" name="kk" min="1" max="25" value="<?= $kk ?>">
    </label>
    <button type="submit">Hitung ulang</button>
  </form>
  <?php if ($errKl !== ''): ?>
    <div class="alert err"><?= e($errKl) ?></div>
  <?php elseif ($klas !== null): ?>
    <div class="kartu-wrap">
      <?= kartu('Akurasi model', fmt($klas['akurasi'] * 100, 2) . '%') ?>
      <?= kartu('Baseline (kelas terbanyak)', fmt($klas['baseline'] * 100, 2) . '%') ?>
      <?= kartu('Data latih', fmt($klas['jumlah_latih'], 0)) ?>
      <?= kartu('Data uji', fmt($klas['jumlah_uji'], 0)) ?>
    </div>
    <div class="alert <?= $klas['akurasi'] > $klas['baseline'] ? 'ok' : 'warn' ?>">
      Akurasi model <?= $klas['akurasi'] > $klas['baseline'] ? 'lebih tinggi' : 'tidak lebih tinggi' ?>
      dari baseline (menebak kelas terbanyak).
    </div>
    <h3>Precision, recall, dan F1 per kelas</h3>
    <?php
    $rowsK = [];
    foreach ($klas['per_kelas'] as $pk) {
        $rowsK[] = [e($pk['kelas']), e(fmt($pk['jumlah'], 0)), e(fmt($pk['precision'], 4)), e(fmt($pk['recall'], 4)), e(fmt($pk['f1'], 4))];
    }
    echo tabel(['Kelas', 'Jumlah (seluruh data)', 'Precision', 'Recall', 'F1'], $rowsK);
    ?>
    <h3>Matriks kebingungan (baris = aktual, kolom = prediksi)</h3>
    <?php
    $head = array_merge(['Aktual \\ Prediksi'], $klas['kelas']);
    $rowsM = [];
    foreach ($klas['matriks'] as $i => $baris) {
        $r = [e($klas['kelas'][$i])];
        foreach ($baris as $v) {
            $r[] = e(fmt($v, 0));
        }
        $rowsM[] = $r;
    }
    echo tabel($head, $rowsM);
    ?>
    <p class="muted">Catatan: kelas yang jumlahnya sedikit biasanya sulit diprediksi, sehingga recall dan F1-nya rendah.
      Coba ubah k untuk melihat pengaruhnya.</p>
  <?php endif; ?>
<?php endif;

if (!$analisis && $menu !== 1):
?>
  <div class="alert warn">Analisis belum tersedia. Periksa dataset di menu 1.</div>
<?php endif; ?>
</div>

<?php if ($charts): ?>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
<?php foreach ($charts as $id => $konfig):
    echo 'new Chart(document.getElementById(' . json_encode($id) . '), '
       . json_encode($konfig, JSON_HEX_TAG | JSON_HEX_AMP | JSON_THROW_ON_ERROR) . ");\n";
endforeach; ?>
</script>
<?php endif; ?>
</body>
</html>
