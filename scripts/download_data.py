"""Scarica i file del dataset Steam (UCSD) nella cartella data/raw/.

Uso, dalla cartella principale del progetto:
    python scripts/download_data.py

I file già presenti non vengono riscaricati. Per ogni file si provano più
indirizzi, perché nel tempo la UCSD li ha spostati.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import GAMES_FILE, RAW_DIR, REVIEWS_FILE, USERS_ITEMS_FILE  # noqa: E402

SOURCES = {
    USERS_ITEMS_FILE: [
        "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_users_items.json.gz",
        "http://jmcauley.ucsd.edu/data/steam/australian_users_items.json.gz",
    ],
    REVIEWS_FILE: [
        "https://mcauleylab.ucsd.edu/public_datasets/data/steam/australian_user_reviews.json.gz",
        "http://jmcauley.ucsd.edu/data/steam/australian_user_reviews.json.gz",
    ],
    GAMES_FILE: [
        "https://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
        "http://cseweb.ucsd.edu/~wckang/steam_games.json.gz",
    ],
}

GZIP_MAGIC = b"\x1f\x8b"  # i primi due byte di ogni file .gz


def download(url: str, dest: Path) -> None:
    """Scarica url in dest mostrando l'avanzamento; scrive prima su un file temporaneo."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MB alla volta
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
            else:
                print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()

    # Un server che risponde con una pagina di errore HTML non produce un .gz valido
    with open(tmp, "rb") as f:
        if f.read(2) != GZIP_MAGIC:
            tmp.unlink()
            raise ValueError("il file scaricato non è un archivio .gz")
    tmp.replace(dest)


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for dest, urls in SOURCES.items():
        if dest.exists():
            print(f"[già presente] {dest.name}")
            continue
        print(f"[download] {dest.name}")
        for url in urls:
            try:
                print(f"  da {url}")
                download(url, dest)
                print(f"  ok ({dest.stat().st_size / 1e6:.1f} MB)")
                break
            except Exception as e:  # noqa: BLE001 - proviamo il prossimo indirizzo
                print(f"  non riuscito: {e}")
        else:
            missing.append(dest.name)

    if missing:
        print("\nNon sono riuscito a scaricare:", ", ".join(missing))
        print("Scaricali a mano dalla pagina https://cseweb.ucsd.edu/~jmcauley/datasets.html")
        print(f"(sezione 'Steam Video Game and Bundle Data') e mettili in {RAW_DIR}")
        return 1

    print("\nTutti i file sono in", RAW_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
