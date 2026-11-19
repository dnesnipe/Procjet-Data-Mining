<?php
declare(strict_types=1);

/**
 * Konfigurasi aplikasi. Dikembalikan sebagai array.
 */
return [
    // Dataset default: file CSV katalog USGS di folder induk (sistem_gempa/)
    'data_default'     => dirname(__DIR__) . DIRECTORY_SEPARATOR . 'Data_Mining_Gempa.csv',
    // Dataset hasil upload (menimpa default bila ada)
    'data_upload'      => __DIR__ . DIRECTORY_SEPARATOR . 'data' . DIRECTORY_SEPARATOR . 'upload_terakhir.csv',
    'upload_maks_byte' => 5 * 1024 * 1024,
    // Fitur clustering (sama dengan versi Python)
    'fitur'            => ['latitude', 'longitude', 'kedalaman_km', 'magnitudo'],
    'k_maks'           => 10,
    'seed'             => 42,
];
