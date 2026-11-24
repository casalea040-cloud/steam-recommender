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

# Tabelle pulite (src/preprocess.py) e divisione training/test (src/split.py)
INTERACTIONS_TABLE = PROCESSED_DIR / "interactions.parquet"  # utente, gioco, ore, gradimento
CLEAN_GAMES_TABLE = PROCESSED_DIR / "games.parquet"          # i giochi rimasti dopo la pulizia
TEST_TABLE = PROCESSED_DIR / "test.parquet"                  # i giochi graditi nascosti per il test

# ---------------------------------------------------------------------------
# Parametri delle scelte del progetto, tutti in un unico punto
# ---------------------------------------------------------------------------

# Indice di gradimento: gioco gradito se giocato almeno 2 ore (finestra di
# rimborso di Steam), salvo recensione, che prevale sulle ore
MIN_PLAYTIME_MINUTES = 120

# Pulizia: soglie minime (provvisorie, da confermare guardando i risultati)
MIN_LIKED_PER_USER = 5     # giochi graditi per tenere un utente
MIN_LIKERS_PER_GAME = 5    # utenti che lo gradiscono per tenere un gioco

# Divisione training/test
TEST_FRACTION = 0.2        # quota dei giochi graditi di ogni utente nascosta per il test
RANDOM_SEED = 42           # stesso seme = stessa divisione a ogni esecuzione
