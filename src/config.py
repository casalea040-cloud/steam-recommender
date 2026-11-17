"""Percorsi del progetto, in un unico punto."""

from pathlib import Path

# Cartella principale del progetto (quella che contiene src/, data/, ...)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # file scaricati, così come sono
PROCESSED_DIR = DATA_DIR / "processed"  # tabelle già pulite (fasi successive)

# Nomi dei file del dataset Steam (UCSD, Julian McAuley)
USERS_ITEMS_FILE = RAW_DIR / "australian_users_items.json.gz"
REVIEWS_FILE = RAW_DIR / "australian_user_reviews.json.gz"
GAMES_FILE = RAW_DIR / "steam_games.json.gz"
