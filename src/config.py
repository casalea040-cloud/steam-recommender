"""Percorsi del progetto, in un unico punto."""

from pathlib import Path

# Cartella principale del progetto (quella che contiene src/, data/, ...)
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # file scaricati, così come sono
INTERIM_DIR = DATA_DIR / "interim"      # gli stessi dati già letti, salvati come tabelle
PROCESSED_DIR = DATA_DIR / "processed"  # tabelle già pulite (fasi successive)

# Nomi dei file del dataset Steam (UCSD, Julian McAuley)
USERS_ITEMS_FILE = RAW_DIR / "australian_users_items.json.gz"
REVIEWS_FILE = RAW_DIR / "australian_user_reviews.json.gz"
GAMES_FILE = RAW_DIR / "steam_games.json.gz"

# Le tre tabelle lette dai file .gz e salvate in formato Parquet,
# per non dover rifare ogni volta la lettura lenta riga per riga
USERS_ITEMS_TABLE = INTERIM_DIR / "users_items.parquet"
GAMES_TABLE = INTERIM_DIR / "games.parquet"
REVIEWS_TABLE = INTERIM_DIR / "reviews.parquet"
