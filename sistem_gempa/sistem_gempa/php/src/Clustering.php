<?php
declare(strict_types=1);

namespace SistemGempa;

use InvalidArgumentException;

/**
 * Tahap 3-5 KDD: normalisasi, K-Means, evaluasi K, profil & kerawanan cluster.
 */
final class Clustering
{
    /**
     * Normalisasi fitur (Z-score atau Min-Max).
     *
     * @return array{0: list<list<float>>, 1: array<string, mixed>}
     */
    public static function skala(array $data, array $fitur, string $metode = 'zscore'): array
    {
        $n = count($data);
        if ($n === 0) {
            throw new InvalidArgumentException('Data kosong.');
        }

        $X = [];
        foreach ($data as $r) {
            $baris = [];
            foreach ($fitur as $f) {
                $baris[] = (float) $r[$f];
            }
            $X[] = $baris;
        }

        $pusat = [];
        $skala = [];
        foreach ($fitur as $j => $_) {
            $kol = array_column($X, $j);
            if ($metode === 'minmax') {
                $lo       = min($kol);
                $rentang  = max($kol) - $lo;
                $pusat[]  = $lo;
                $skala[]  = $rentang > 0 ? $rentang : 1.0;
            } else {
                $mean = array_sum($kol) / $n;
                $var  = 0.0;
                foreach ($kol as $v) {
                    $var += ($v - $mean) ** 2;
                }
                $sd      = sqrt($var / $n);
                $pusat[] = $mean;
                $skala[] = $sd > 0 ? $sd : 1.0;
            }
        }

        foreach ($X as $i => $baris) {
            foreach ($baris as $j => $v) {
                $X[$i][$j] = ($v - $pusat[$j]) / $skala[$j];
            }
        }

        return [$X, ['metode' => $metode, 'fitur' => array_values($fitur), 'pusat' => $pusat, 'skala' => $skala]];
    }

    /** Terapkan parameter skala latih ke nilai baru (untuk prediksi). */
    public static function terapkanSkala(array $nilai, array $fitur, array $params): array
    {
        $x = [];
        foreach ($fitur as $j => $f) {
            $x[] = ((float) $nilai[$f] - $params['pusat'][$j]) / $params['skala'][$j];
        }
        return $x;
    }

    /**
     * K-Means (k-means++ init, n_init kali, pilih SSE terkecil).
     *
     * @return array{labels: list<int>, centroids: list<list<float>>, sse: float}
     */
    public static function kmeans(array $X, int $k, int $nInit = 20, int $seed = 42, int $maksIter = 300): array
    {
        if ($k < 1 || $k > count($X)) {
            throw new InvalidArgumentException('K harus antara 1 dan jumlah data.');
        }
        mt_srand($seed);
        $terbaik = null;
        for ($r = 0; $r < $nInit; $r++) {
            $run = self::kmeansSekali($X, $k, $maksIter);
            if ($terbaik === null || $run['sse'] < $terbaik['sse']) {
                $terbaik = $run;
            }
        }
        return $terbaik;
    }

    private static function kmeansSekali(array $X, int $k, int $maksIter): array
    {
        $cent  = self::inisialisasiPlusPlus($X, $k);
        $label = self::tugaskan($X, $cent);

        for ($it = 0; $it < $maksIter; $it++) {
            $cent  = self::hitungPusat($X, $label, $cent);
            $baru  = self::tugaskan($X, $cent);
            if ($baru === $label) {
                break;
            }
            $label = $baru;
        }

        $sse = 0.0;
        foreach ($X as $i => $x) {
            $sse += self::jarak2($x, $cent[$label[$i]]);
        }
        return ['labels' => $label, 'centroids' => $cent, 'sse' => $sse];
    }

    private static function inisialisasiPlusPlus(array $X, int $k): array
    {
        $n    = count($X);
        $cent = [$X[mt_rand(0, $n - 1)]];
        $d2   = array_fill(0, $n, INF);

        while (count($cent) < $k) {
            $terakhir = end($cent);
            $total    = 0.0;
            foreach ($X as $i => $x) {
                $d2[$i] = min($d2[$i], self::jarak2($x, $terakhir));
                $total += $d2[$i];
            }
            if ($total <= 0) {
                $cent[] = $X[mt_rand(0, $n - 1)];
                continue;
            }
            $r     = (mt_rand() / mt_getrandmax()) * $total;
            $acc   = 0.0;
            $pilih = $n - 1;
            foreach ($d2 as $i => $d) {
                $acc += $d;
                if ($acc >= $r) {
                    $pilih = $i;
                    break;
                }
            }
            $cent[] = $X[$pilih];
        }
        return $cent;
    }

    private static function tugaskan(array $X, array $cent): array
    {
        $label = [];
        foreach ($X as $x) {
            $best = 0;
            $bd   = INF;
            foreach ($cent as $c => $pc) {
                $d = self::jarak2($x, $pc);
                if ($d < $bd) {
                    $bd   = $d;
                    $best = $c;
                }
            }
            $label[] = $best;
        }
        return $label;
    }

    private static function hitungPusat(array $X, array $label, array $cent): array
    {
        $k = count($cent);
        $d = count($X[0]);
        $sum = array_fill(0, $k, array_fill(0, $d, 0.0));
        $jml = array_fill(0, $k, 0);

        foreach ($X as $i => $x) {
            $c = $label[$i];
            $jml[$c]++;
            for ($j = 0; $j < $d; $j++) {
                $sum[$c][$j] += $x[$j];
            }
        }

        // jarak tiap titik ke pusat miliknya, untuk menangani cluster kosong
        $jarak = [];
        foreach ($X as $i => $x) {
            $jarak[$i] = self::jarak2($x, $cent[$label[$i]]);
        }

        for ($c = 0; $c < $k; $c++) {
            if ($jml[$c] > 0) {
                for ($j = 0; $j < $d; $j++) {
                    $cent[$c][$j] = $sum[$c][$j] / $jml[$c];
                }
            } else {
                // cluster kosong: pindahkan ke titik yang paling jauh dari pusatnya
                $far = array_keys($jarak, max($jarak), true)[0];
                $cent[$c] = $X[$far];
                $jarak[$far] = -1.0;
            }
        }
        return $cent;
    }

    private static function jarak2(array $a, array $b): float
    {
        $s = 0.0;
        foreach ($a as $j => $v) {
            $t  = $v - $b[$j];
            $s += $t * $t;
        }
        return $s;
    }

    /** Silhouette score rata-rata (-1..1). */
    public static function silhouette(array $X, array $labels, int $k): float
    {
        $n   = count($X);
        $jml = array_fill(0, $k, 0);
        foreach ($labels as $l) {
            $jml[$l]++;
        }
        if (count(array_filter($jml)) < 2) {
            return 0.0;
        }

        $total = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $own = $labels[$i];
            if ($jml[$own] <= 1) {
                continue; // s = 0
            }
            $sum = array_fill(0, $k, 0.0);
            for ($j = 0; $j < $n; $j++) {
                if ($j !== $i) {
                    $sum[$labels[$j]] += sqrt(self::jarak2($X[$i], $X[$j]));
                }
            }
            $a = $sum[$own] / ($jml[$own] - 1);
            $b = INF;
            for ($c = 0; $c < $k; $c++) {
                if ($c !== $own && $jml[$c] > 0) {
                    $b = min($b, $sum[$c] / $jml[$c]);
                }
            }
            $m = max($a, $b);
            if ($m > 0) {
                $total += ($b - $a) / $m;
            }
        }
        return $total / $n;
    }

    /** Davies-Bouldin Index (makin rendah makin baik). */
    public static function daviesBouldin(array $X, array $labels, array $cent): float
    {
        $k   = count($cent);
        $S   = array_fill(0, $k, 0.0);
        $jml = array_fill(0, $k, 0);
        foreach ($X as $i => $x) {
            $c = $labels[$i];
            $S[$c] += sqrt(self::jarak2($x, $cent[$c]));
            $jml[$c]++;
        }
        for ($c = 0; $c < $k; $c++) {
            if ($jml[$c] > 0) {
                $S[$c] /= $jml[$c];
            }
        }

        $total = 0.0;
        $aktif = 0;
        for ($i = 0; $i < $k; $i++) {
            if ($jml[$i] === 0) {
                continue;
            }
            $aktif++;
            $maks = 0.0;
            for ($j = 0; $j < $k; $j++) {
                if ($j === $i || $jml[$j] === 0) {
                    continue;
                }
                $m = sqrt(self::jarak2($cent[$i], $cent[$j]));
                if ($m > 0) {
                    $maks = max($maks, ($S[$i] + $S[$j]) / $m);
                }
            }
            $total += $maks;
        }
        return $aktif > 0 ? $total / $aktif : 0.0;
    }

    /** Evaluasi K = kMin..kMax: SSE, Silhouette, Davies-Bouldin. */
    public static function evaluasiK(array $X, int $kMin = 2, int $kMax = 10): array
    {
        $rows = [];
        for ($k = $kMin; $k <= $kMax; $k++) {
            $run    = self::kmeans($X, $k);
            $rows[] = [
                'K'              => $k,
                'SSE'            => round($run['sse'], 4),
                'Silhouette'     => round(self::silhouette($X, $run['labels'], $k), 4),
                'Davies_Bouldin' => round(self::daviesBouldin($X, $run['labels'], $run['centroids']), 4),
            ];
        }
        return $rows;
    }

    /** K dengan Silhouette tertinggi. */
    public static function pilihK(array $ev): int
    {
        $best = $ev[0];
        foreach ($ev as $r) {
            if ($r['Silhouette'] > $best['Silhouette']) {
                $best = $r;
            }
        }
        return (int) $best['K'];
    }

    /** Titik siku Elbow: titik terjauh dari garis antara titik awal dan akhir kurva SSE. */
    public static function titikSiku(array $ev): int
    {
        $n = count($ev);
        if ($n < 3) {
            return (int) $ev[0]['K'];
        }
        $kMin = $ev[0]['K'];
        $kMax = $ev[$n - 1]['K'];
        $sse  = array_column($ev, 'SSE');
        $sMin = min($sse);
        $sMax = max($sse);

        $xs = [];
        $ys = [];
        foreach ($ev as $r) {
            $xs[] = ($r['K'] - $kMin) / max(1, $kMax - $kMin);
            $ys[] = $sMax > $sMin ? ($r['SSE'] - $sMin) / ($sMax - $sMin) : 0.0;
        }

        $dx  = $xs[$n - 1] - $xs[0];
        $dy  = $ys[$n - 1] - $ys[0];
        $len = sqrt($dx * $dx + $dy * $dy);
        $best = 0;
        $bd   = -1.0;
        for ($i = 0; $i < $n; $i++) {
            $d = $len > 0 ? abs($dy * ($xs[$i] - $xs[0]) - $dx * ($ys[$i] - $ys[0])) / $len : 0.0;
            if ($d > $bd) {
                $bd   = $d;
                $best = $i;
            }
        }
        return (int) $ev[$best]['K'];
    }

    /**
     * Jalankan K-Means lengkap: hasil per gempa, profil cluster, metrik, parameter skala.
     * Cluster diberi nomor 1..K berdasarkan jumlah anggota (terbanyak = 1).
     */
    public static function jalankan(array $data, array $fitur, string $metode, int $k): array
    {
        [$X, $params] = self::skala($data, $fitur, $metode);
        $run   = self::kmeans($X, $k);
        $label = $run['labels'];
        $cent  = $run['centroids'];

        $jml = array_fill(0, $k, 0);
        foreach ($label as $l) {
            $jml[$l]++;
        }
        $urut = range(0, $k - 1);
        usort($urut, static fn ($a, $b) => [$jml[$b], $a] <=> [$jml[$a], $b]);

        $peta     = [];
        $centroid = [];
        foreach ($urut as $i => $lama) {
            $peta[$lama]   = $i + 1;
            $centroid[$i]  = $cent[$lama];
        }

        $hasil = [];
        $labelBaru = [];
        foreach ($data as $i => $r) {
            $r['cluster'] = $peta[$label[$i]];
            $hasil[]      = $r;
            $labelBaru[]  = $label[$i];
        }
        $labelIdx = array_map(static fn ($l) => $peta[$l] - 1, $labelBaru);

        $metrik = [
            'Silhouette'     => round(self::silhouette($X, $labelIdx, $k), 4),
            'Davies_Bouldin' => round(self::daviesBouldin($X, $labelIdx, $centroid), 4),
            'SSE'            => round($run['sse'], 4),
        ];

        return [
            'hasil'    => $hasil,
            'profil'   => self::profil($hasil),
            'metrik'   => $metrik,
            'params'   => $params,
            'centroid' => $centroid,
        ];
    }

    /** Profil tiap cluster beserta skor & tingkat kerawanan. */
    public static function profil(array $hasil): array
    {
        $total    = count($hasil);
        $kelompok = [];
        foreach ($hasil as $r) {
            $kelompok[$r['cluster']][] = $r;
        }
        ksort($kelompok);

        $stat = [];
        foreach ($kelompok as $c => $anggota) {
            $n   = count($anggota);
            $dep = array_column($anggota, 'kedalaman_km');
            $mag = array_column($anggota, 'magnitudo');
            $wil = array_count_values(array_column($anggota, 'wilayah_acuan'));
            arsort($wil);
            $dangkal = count(array_filter($dep, static fn ($d) => $d < 70));

            $stat[$c] = [
                'cluster'            => $c,
                'jumlah_gempa'       => $n,
                'persen'             => round($n / $total * 100, 2),
                'lat_rata2'          => round(self::rata(array_column($anggota, 'latitude')), 4),
                'lon_rata2'          => round(self::rata(array_column($anggota, 'longitude')), 4),
                'kedalaman_rata2_km' => round(self::rata($dep), 1),
                'kedalaman_min_km'   => round(min($dep), 1),
                'kedalaman_maks_km'  => round(max($dep), 1),
                'mag_rata2'          => round(self::rata($mag), 2),
                'mag_maks'           => round(max($mag), 1),
                'wilayah_dominan'    => (string) array_key_first($wil),
                'proporsi_dangkal'   => round($dangkal / $n, 4),
            ];
        }

        $frek = array_column($stat, 'jumlah_gempa');
        $magR = array_column($stat, 'mag_rata2');
        $fMin = min($frek);
        $fMax = max($frek);
        $mMin = min($magR);
        $mMax = max($magR);

        $out = [];
        foreach ($stat as $c => $s) {
            $fn    = self::normal($s['jumlah_gempa'], $fMin, $fMax);
            $mn    = self::normal($s['mag_rata2'], $mMin, $mMax);
            $skor  = round(0.40 * $fn + 0.35 * $mn + 0.25 * $s['proporsi_dangkal'], 4);

            $s['skor_kerawanan']    = $skor;
            $s['tingkat_kerawanan'] = $skor > 0.65 ? 'Tinggi' : ($skor >= 0.40 ? 'Sedang' : 'Rendah');
            $s['karakteristik']     = self::karakteristik($s);
            $out[] = $s;
        }
        return $out;
    }

    private static function karakteristik(array $s): string
    {
        $d = $s['kedalaman_rata2_km'];
        $kat = match (true) {
            $d < 70   => 'Dangkal',
            $d < 300  => 'Menengah',
            default   => 'Dalam',
        };
        return sprintf('%s (±%d km), sekitar %s, magnitudo rata-rata M%.1f',
            $kat, (int) round($d), $s['wilayah_dominan'], $s['mag_rata2']);
    }

    /** Statistik deskriptif (mirip pandas describe). */
    public static function deskriptif(array $data, array $fitur): array
    {
        $out = [];
        foreach ($fitur as $f) {
            $v = array_map('floatval', array_column($data, $f));
            sort($v);
            $n    = count($v);
            $mean = array_sum($v) / $n;
            $var  = 0.0;
            foreach ($v as $x) {
                $var += ($x - $mean) ** 2;
            }
            $out[$f] = [
                'jumlah' => $n,
                'rata2'  => $mean,
                'std'    => $n > 1 ? sqrt($var / ($n - 1)) : 0.0,
                'min'    => $v[0],
                'p25'    => self::persentil($v, 0.25),
                'p50'    => self::persentil($v, 0.50),
                'p75'    => self::persentil($v, 0.75),
                'maks'   => $v[$n - 1],
            ];
        }
        return $out;
    }

    private static function persentil(array $v, float $p): float
    {
        $n   = count($v);
        $pos = ($n - 1) * $p;
        $lo  = (int) floor($pos);
        $hi  = min($lo + 1, $n - 1);
        return $v[$lo] + ($v[$hi] - $v[$lo]) * ($pos - $lo);
    }

    /** Indeks cluster (1..K) dari titik baru, berdasarkan centroid terdekat. */
    public static function prediksi(array $nilai, array $fitur, array $params, array $centroid): int
    {
        $x    = self::terapkanSkala($nilai, $fitur, $params);
        $best = 0;
        $bd   = INF;
        foreach ($centroid as $c => $pc) {
            $d = self::jarak2($x, $pc);
            if ($d < $bd) {
                $bd   = $d;
                $best = $c;
            }
        }
        return $best + 1;
    }

    private static function rata(array $v): float
    {
        return count($v) > 0 ? array_sum($v) / count($v) : 0.0;
    }

    private static function normal(float $v, float $lo, float $hi): float
    {
        return $hi > $lo ? ($v - $lo) / ($hi - $lo) : 0.0;
    }
}
