<?php
declare(strict_types=1);

namespace SistemGempa;

use InvalidArgumentException;

/**
 * Klasifikasi k-Nearest Neighbors (k-NN) dengan evaluasi hold-out.
 *
 * Target default: kelas kedalaman (Dangkal / Menengah / Dalam).
 * Fitur: latitude, longitude, magnitudo (kedalaman tidak dipakai sebagai fitur
 * karena menjadi target).
 */
final class Klasifikasi
{
    /**
     * @return array<string, mixed>
     */
    public static function knn(
        array $data,
        array $fitur,
        string $target,
        int $k = 5,
        float $proporsiUji = 0.3,
        int $seed = 42
    ): array {
        $n = count($data);
        if ($n < 10) {
            throw new InvalidArgumentException('Data terlalu sedikit untuk klasifikasi (minimal 10).');
        }
        if ($k < 1 || $k >= $n) {
            throw new InvalidArgumentException('Nilai k tidak valid.');
        }

        // Pembagian data latih/uji yang dapat diulang (seed tetap)
        mt_srand($seed);
        $urut = range(0, $n - 1);
        shuffle($urut);
        $jumlahUji = max(1, (int) round($n * $proporsiUji));
        $uji   = array_slice($urut, 0, $jumlahUji);
        $latih = array_slice($urut, $jumlahUji);

        // Standarisasi memakai statistik data latih saja (hindari kebocoran data)
        $mean = [];
        $sd   = [];
        foreach ($fitur as $f) {
            $v = array_map(static fn ($i) => (float) $data[$i][$f], $latih);
            $m = array_sum($v) / count($v);
            $var = 0.0;
            foreach ($v as $x) {
                $var += ($x - $m) ** 2;
            }
            $mean[$f] = $m;
            $sd[$f]   = sqrt($var / count($v)) ?: 1.0;
        }

        $X = [];
        foreach ($data as $i => $r) {
            $baris = [];
            foreach ($fitur as $f) {
                $baris[] = ((float) $r[$f] - $mean[$f]) / $sd[$f];
            }
            $X[$i] = $baris;
        }

        $kelas = array_values(array_unique(array_column($data, $target)));
        sort($kelas);

        // Prediksi tiap data uji dengan voting k tetangga terdekat
        $cm    = [];
        $benar = 0;
        foreach ($uji as $i) {
            $jarak = [];
            foreach ($latih as $j) {
                $jarak[] = [self::jarak2($X[$i], $X[$j]), $data[$j][$target]];
            }
            usort($jarak, static fn ($a, $b) => $a[0] <=> $b[0]);

            $suara = [];
            foreach (array_slice($jarak, 0, $k) as [, $c]) {
                $suara[$c] = ($suara[$c] ?? 0) + 1;
            }
            arsort($suara); // seri: kelas dari tetangga terdekat didahulukan
            $prediksi = (string) array_key_first($suara);
            $aktual   = (string) $data[$i][$target];

            $cm[$aktual][$prediksi] = ($cm[$aktual][$prediksi] ?? 0) + 1;
            if ($prediksi === $aktual) {
                $benar++;
            }
        }

        $hitung = array_count_values(array_column($data, $target));

        // Metrik per kelas
        $perKelas = [];
        foreach ($kelas as $c) {
            $tp = $cm[$c][$c] ?? 0;
            $fp = 0;
            $fn = 0;
            foreach ($kelas as $o) {
                if ($o !== $c) {
                    $fp += $cm[$o][$c] ?? 0;
                    $fn += $cm[$c][$o] ?? 0;
                }
            }
            $prec = $tp + $fp > 0 ? $tp / ($tp + $fp) : 0.0;
            $rec  = $tp + $fn > 0 ? $tp / ($tp + $fn) : 0.0;
            $f1   = $prec + $rec > 0 ? 2 * $prec * $rec / ($prec + $rec) : 0.0;
            $perKelas[] = [
                'kelas'     => $c,
                'jumlah'    => $hitung[$c] ?? 0,
                'precision' => round($prec, 4),
                'recall'    => round($rec, 4),
                'f1'        => round($f1, 4),
            ];
        }

        // Matriks kebingungan: baris = aktual, kolom = prediksi
        $matriks = [];
        foreach ($kelas as $a) {
            $baris = [];
            foreach ($kelas as $p) {
                $baris[] = $cm[$a][$p] ?? 0;
            }
            $matriks[] = $baris;
        }

        // Baseline: akurasi jika selalu menebak kelas terbanyak
        $baseline = max($hitung) / $n;

        return [
            'target'       => $target,
            'fitur'        => $fitur,
            'k'            => $k,
            'proporsi_uji' => $proporsiUji,
            'jumlah_latih' => count($latih),
            'jumlah_uji'   => count($uji),
            'akurasi'      => round($benar / count($uji), 4),
            'baseline'     => round($baseline, 4),
            'kelas'        => $kelas,
            'per_kelas'    => $perKelas,
            'matriks'      => $matriks,
        ];
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
}
