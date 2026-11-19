<?php
declare(strict_types=1);

namespace SistemGempa;

use RuntimeException;

/**
 * Tahap 1-2 KDD: selection & preprocessing data katalog USGS (CSV).
 *
 * Mendukung dua format:
 *  - CSV USGS asli (pemisah koma, desimal titik)
 *  - CSV hasil ekspor Excel Indonesia (pemisah titik koma, tanda desimal hilang),
 *    tanda desimalnya dipulihkan otomatis.
 */
final class Preprocessing
{
    private const LAT_MIN = -4.0;
    private const LAT_MAX = 3.0;
    private const LON_MIN = 118.0;
    private const LON_MAX = 126.0;
    private const DEPTH_MAKS = 700.0;
    private const KNN_K = 5;

    /** Koordinat kota acuan pada kolom place (untuk memilih latitude yang benar). */
    private const KOTA = [
        'Palu'    => [-0.8986, 119.8707],
        'Poso'    => [-1.3950, 120.7520],
        'Luwuk'   => [-0.9430, 122.7860],
        'Masamba' => [-2.5270, 120.3100],
        'Mamuju'  => [-2.6790, 118.8890],
    ];

    /**
     * Membaca CSV dan membersihkannya.
     *
     * @return array{0: list<array<string, mixed>>, 1: array<string, int|string>}
     */
    public static function muatDanBersihkan(string $path, bool $tanpaGorontalo = true): array
    {
        $fh = @fopen($path, 'rb');
        if ($fh === false) {
            throw new RuntimeException('File tidak dapat dibuka: ' . basename($path));
        }

        // Deteksi pemisah dari baris pertama (header)
        $baris1 = (string) fgets($fh);
        rewind($fh);
        $sep = substr_count($baris1, ';') > substr_count($baris1, ',') ? ';' : ',';

        $header = null;
        $baris  = [];
        while (($row = fgetcsv($fh, 0, $sep)) !== false) {
            if ($row === [null]) {
                continue; // baris kosong
            }
            if ($header === null) {
                $row[0] = preg_replace('/^\xEF\xBB\xBF/', '', (string) $row[0]); // buang BOM
                $header = array_map(static fn ($h) => trim((string) $h), $row);
                continue;
            }
            $baris[] = $row;
        }
        fclose($fh);

        if ($header === null) {
            throw new RuntimeException('File CSV kosong.');
        }
        $idx = array_flip($header); // nama kolom => posisi

        $lap = [
            'pemisah_kolom'             => $sep === ';' ? 'titik koma (;)' : 'koma (,)',
            'baris_pada_file'           => count($baris),
            'baris_kosong_dibuang'      => 0,
            'duplikat_dibuang'          => 0,
            'gorontalo_dikecualikan'    => 0,
            'koordinat_di_luar_wilayah' => 0,
            'baris_desimal_dipulihkan'  => 0,
        ];

        // Pass 1: rekonstruksi bujur, lintang, magnitudo; kumpulkan kandidat kedalaman
        $pre = [];
        foreach ($baris as $r) {
            $lon = self::bujur(self::kolom($r, $idx, 'longitude'));
            $place = (string) self::kolom($r, $idx, 'place');
            $pre[] = [
                'id'     => (string) self::kolom($r, $idx, 'id'),
                'waktu'  => (string) self::kolom($r, $idx, 'time'),
                'place'  => $place,
                'lon'    => $lon,
                'lat'    => self::lintang(self::kolom($r, $idx, 'latitude'), $place, $lon[0]),
                'mag'    => self::magnitudo(self::kolom($r, $idx, 'mag')),
                'depRaw' => self::kolom($r, $idx, 'depth'),
                'tipe'   => (string) self::kolom($r, $idx, 'magType'),
            ];
        }

        // Titik acuan kedalaman: yang nilainya sudah pasti (tanpa ambiguitas)
        $ref = [];
        foreach ($pre as $p) {
            if ($p['lat'][0] === null || $p['lon'][0] === null) {
                continue;
            }
            $kand = self::kedalamanKandidat($p['depRaw']);
            if (count($kand) === 1) {
                $ref[] = ['lat' => $p['lat'][0], 'lon' => $p['lon'][0], 'd' => $kand[0]];
            }
        }

        // Pass 2: seleksi, validasi, dan penentuan kedalaman
        $data     = [];
        $sudahAda = [];
        foreach ($pre as $p) {
            $dep = self::kedalaman($p['depRaw'], $p['lat'][0], $p['lon'][0], $ref);

            if ($p['lat'][0] === null || $p['lon'][0] === null || $dep[0] === null || $p['mag'][0] === null) {
                $lap['baris_kosong_dibuang']++;
                continue;
            }

            $id = $p['id'];
            if ($id !== '') {
                if (isset($sudahAda[$id])) {
                    $lap['duplikat_dibuang']++;
                    continue;
                }
                $sudahAda[$id] = true;
            }

            if ($tanpaGorontalo && stripos($p['place'], 'Gorontalo') !== false) {
                $lap['gorontalo_dikecualikan']++;
                continue;
            }

            $lat = $p['lat'][0];
            $lon = $p['lon'][0];
            if ($lat < self::LAT_MIN || $lat > self::LAT_MAX
                || $lon < self::LON_MIN || $lon > self::LON_MAX || $dep[0] < 0) {
                $lap['koordinat_di_luar_wilayah']++;
                continue;
            }

            if ($p['lat'][1] || $p['lon'][1] || $dep[1] || $p['mag'][1]) {
                $lap['baris_desimal_dipulihkan']++;
            }

            $tahun = preg_match('/(\d{4})/', $p['waktu'], $m) ? (int) $m[1] : 0;
            $data[] = [
                'id'              => $id,
                'waktu'           => $p['waktu'],
                'tahun'           => $tahun,
                'latitude'        => $lat,
                'longitude'       => $lon,
                'kedalaman_km'    => $dep[0],
                'magnitudo'       => $p['mag'][0],
                'jenis_magnitudo' => $p['tipe'],
                'lokasi'          => $p['place'],
                'wilayah_acuan'   => self::wilayahAcuan($p['place']),
                'kelas_kedalaman' => self::kelasKedalaman($dep[0]),
                'kelas_magnitudo' => self::kelasMagnitudo($p['mag'][0]),
            ];
        }

        $lap['baris_bersih'] = count($data);
        $tahun = array_filter(array_column($data, 'tahun'));
        $lap['periode'] = $tahun ? min($tahun) . ' – ' . max($tahun) : '-';
        $lap['catatan'] = $lap['baris_desimal_dipulihkan'] > 0
            ? 'Tanda desimal dipulihkan otomatis untuk ' . $lap['baris_desimal_dipulihkan']
              . ' baris. Latitude dipilih dengan jarak ke kota acuan di kolom place; '
              . 'kedalaman dipilih dari median ' . self::KNN_K . ' gempa terdekat (perkiraan).'
            : 'Tanda desimal lengkap, tidak perlu rekonstruksi.';

        return [$data, $lap];
    }

    /** Ambil nilai kolom sebagai string (null bila kolom tidak ada). */
    private static function kolom(array $r, array $idx, string $nama): ?string
    {
        if (!isset($idx[$nama])) {
            return null;
        }
        $v = $r[$idx[$nama]] ?? null;
        return $v === null ? null : trim((string) $v);
    }

    /**
     * Bujur: 3 digit bilangan bulat lalu desimal, mis. 1215669 -> 121.5669.
     * @return array{0: ?float, 1: bool} nilai dan apakah direkonstruksi
     */
    private static function bujur(?string $raw): array
    {
        if ($raw === null || $raw === '' || !is_numeric(ltrim($raw, '+-'))) {
            return [null, false];
        }
        if (str_contains($raw, '.')) {
            return [(float) $raw, false];
        }
        $neg    = str_starts_with($raw, '-');
        $digits = ltrim($raw, '+-');
        if (strlen($digits) <= 3) {
            return [($neg ? -1 : 1) * (float) $digits, false];
        }
        $v = (float) (substr($digits, 0, 3) . '.' . substr($digits, 3));
        return [$neg ? -$v : $v, true];
    }

    /**
     * Latitude: posisi desimalnya ambigu (-164 = -1.64 atau -0.164).
     * Dipilih kandidat yang jaraknya ke kota acuan paling sesuai dengan keterangan
     * "N km ARAH of KOTA" pada kolom place. Tanpa keterangan, dipilih yang paling dekat -1.
     * @return array{0: ?float, 1: bool}
     */
    private static function lintang(?string $raw, string $place, ?float $lon): array
    {
        if ($raw === null || $raw === '' || !is_numeric(ltrim($raw, '+-'))) {
            return [null, false];
        }
        if (str_contains($raw, '.')) {
            return [(float) $raw, false];
        }
        $neg    = str_starts_with($raw, '-');
        $digits = ltrim($raw, '+-');
        if ((float) $digits === 0.0) {
            return [0.0, false];
        }

        $kandidat = [];
        for ($d = 1; $d <= 4; $d++) {
            $v = (float) $digits / (10 ** $d);
            $v = $neg ? -$v : $v;
            if ($v >= self::LAT_MIN && $v <= self::LAT_MAX) {
                $kandidat[] = $v;
            }
        }
        if (!$kandidat) {
            return [null, false];
        }
        if (count($kandidat) === 1) {
            return [$kandidat[0], true];
        }

        $kota = self::jarakKota($place);
        if ($kota !== null && $lon !== null) {
            [$km, $nama] = $kota;
            [$kLat, $kLon] = self::KOTA[$nama];
            usort($kandidat, static fn ($a, $b) =>
                abs(self::haversine($a, $lon, $kLat, $kLon) - $km)
                <=> abs(self::haversine($b, $lon, $kLat, $kLon) - $km));
            return [$kandidat[0], true];
        }

        usort($kandidat, static fn ($a, $b) => abs($a + 1.0) <=> abs($b + 1.0));
        return [$kandidat[0], true];
    }

    /**
     * Magnitudo: "45" -> 4.5. Nilai satu digit dianggap bilangan bulat.
     * @return array{0: ?float, 1: bool}
     */
    private static function magnitudo(?string $raw): array
    {
        if ($raw === null || $raw === '' || !is_numeric($raw)) {
            return [null, false];
        }
        if (str_contains($raw, '.') || strlen($raw) < 2) {
            return [(float) $raw, false];
        }
        return [(float) ($raw[0] . '.' . substr($raw, 1)), true];
    }

    /**
     * Kandidat kedalaman (km). Angka pendek (<= 3 digit) dianggap sudah dalam km.
     * Angka lebih panjang bisa berarti 1-3 desimal; hanya kandidat <= 700 km yang diambil.
     * @return list<float>
     */
    private static function kedalamanKandidat(?string $raw): array
    {
        if ($raw === null || $raw === '' || !is_numeric(ltrim($raw, '+-'))) {
            return [];
        }
        if (str_contains($raw, '.')) {
            return [(float) $raw];
        }
        $digits = ltrim($raw, '+-');
        if (strlen($digits) <= 3 && (float) $digits <= self::DEPTH_MAKS) {
            return [(float) $digits];
        }
        $kand = [];
        for ($d = 1; $d <= 3; $d++) {
            $v = (float) $digits / (10 ** $d);
            if ($v <= self::DEPTH_MAKS) {
                $kand[] = $v;
            }
        }
        return $kand;
    }

    /**
     * Kedalaman final: bila ambigu, pilih kandidat yang paling dekat dengan median
     * kedalaman SKN (5 gempa terdekat yang kedalamannya pasti).
     * @param list<array{lat: float, lon: float, d: float}> $ref
     * @return array{0: ?float, 1: bool}
     */
    private static function kedalaman(?string $raw, ?float $lat, ?float $lon, array $ref): array
    {
        $kand = self::kedalamanKandidat($raw);
        if (!$kand) {
            return [null, false];
        }
        $tanpaAmbigu = count($kand) === 1 && (str_contains((string) $raw, '.') || strlen(ltrim((string) $raw, '+-')) <= 3);
        if ($tanpaAmbigu) {
            return [$kand[0], false];
        }
        if (count($kand) === 1) {
            return [$kand[0], true];
        }

        $median = 10.0;
        if ($lat !== null && $lon !== null && $ref) {
            $jarak = [];
            foreach ($ref as $t) {
                $jarak[] = [self::haversine($lat, $lon, $t['lat'], $t['lon']), $t['d']];
            }
            usort($jarak, static fn ($a, $b) => $a[0] <=> $b[0]);
            $dekat = array_column(array_slice($jarak, 0, self::KNN_K), 1);
            sort($dekat);
            $n = count($dekat);
            $median = $n % 2 ? $dekat[intdiv($n, 2)] : ($dekat[$n / 2 - 1] + $dekat[$n / 2]) / 2;
        }

        usort($kand, static fn ($a, $b) => abs($a - $median) <=> abs($b - $median));
        return [$kand[0], true];
    }

    /** "51 km SE of Palu, Indonesia" -> [51.0, "Palu"], jika kota dikenal. */
    private static function jarakKota(string $place): ?array
    {
        if (preg_match('/(\d+(?:\.\d+)?)\s*km\s+[NSEW]{1,3}\s+of\s+([A-Za-z]+)/i', $place, $m)
            && isset(self::KOTA[$m[2]])) {
            return [(float) $m[1], $m[2]];
        }
        return null;
    }

    private static function haversine(float $lat1, float $lon1, float $lat2, float $lon2): float
    {
        $dLat = deg2rad($lat2 - $lat1);
        $dLon = deg2rad($lon2 - $lon1);
        $a    = sin($dLat / 2) ** 2 + cos(deg2rad($lat1)) * cos(deg2rad($lat2)) * sin($dLon / 2) ** 2;
        return 2 * 6371 * asin(min(1.0, sqrt($a)));
    }

    /** "122 km WSW of Gorontalo, Indonesia" -> "Gorontalo" */
    public static function wilayahAcuan(string $place): string
    {
        if (preg_match('/\bof\s+(.+)$/i', $place, $m)) {
            $w = trim(explode(',', $m[1])[0]);
            return $w !== '' ? $w : 'Lainnya';
        }
        return 'Lainnya';
    }

    public static function kelasKedalaman(float $d): string
    {
        return match (true) {
            $d < 70   => 'Dangkal (<70 km)',
            $d <= 300 => 'Menengah (70–300 km)',
            default   => 'Dalam (>300 km)',
        };
    }

    public static function kelasMagnitudo(float $m): string
    {
        return match (true) {
            $m < 4.5  => 'Kecil (<4,5)',
            $m < 5.5  => 'Sedang (4,5–5,4)',
            default   => 'Besar (≥5,5)',
        };
    }
}
