"""Lettura dei file grezzi del dataset Steam e conversione in tabelle pandas.

I file non sono JSON standard: ogni riga è un dizionario Python (apici singoli,
True/False...). Per questo si leggono riga per riga con ast.literal_eval, che
interpreta solo dati e non esegue codice (a differenza di eval).

Funzioni principali:
    load_users_items()  -> una riga per coppia (utente, gioco) con le ore di gioco
    load_games()        -> una riga per gioco con generi, tag, specs, ...
    load_reviews()      -> una riga per recensione, con il voto consigliato sì/no

Per un controllo veloce, dalla cartella principale del progetto:
    python -m src.load
"""

import ast
import gzip
import json
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src.config import GAMES_FILE, REVIEWS_FILE, USERS_ITEMS_FILE


# ---------------------------------------------------------------------------
# Lettura riga per riga
# ---------------------------------------------------------------------------

def read_records(path: Path, limit: Optional[int] = None) -> Iterator[dict]:
    """Legge un file .json.gz e restituisce un dizionario per ogni riga.

    Prova prima ast.literal_eval (formato Python), poi json.loads (JSON vero).
    Le righe illeggibili vengono saltate e contate; il totale viene stampato
    alla fine, così non passano inosservate.

    limit: se indicato, si ferma dopo quel numero di righe (utile per le prove).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} non trovato. Lancia prima: python scripts/download_data.py"
        )

    bad = 0
    n = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
            yield record
            n += 1
            if limit is not None and n >= limit:
                break

    if bad:
        print(f"[attenzione] {path.name}: {bad} righe illeggibili saltate")


def _as_list(value) -> list:
    """Generi, tag e specs a volte mancano: li normalizziamo sempre a lista."""
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [value]


# ---------------------------------------------------------------------------
# Librerie degli utenti (per il collaborative filtering)
# ---------------------------------------------------------------------------

def load_users_items(path: Path = USERS_ITEMS_FILE, limit: Optional[int] = None) -> pd.DataFrame:
    """Una riga per ogni gioco nella libreria di ogni utente.

    Colonne: user_id, item_id, item_name, playtime_forever, playtime_2weeks.
    Le ore sono in minuti, come nell'API di Steam.

    Il file originale ha un record per utente con una lista annidata di giochi;
    qui la "srotoliamo" in formato lungo, il più comodo per tutto il resto.
    Le colonne vengono costruite come liste semplici: con milioni di righe è
    molto più veloce e leggero che creare un dizionario per riga.

    limit: numero massimo di utenti da leggere.
    """
    user_ids, item_ids, names, total, recent = [], [], [], [], []

    for rec in read_records(path, limit=limit):
        uid = str(rec.get("user_id"))
        for it in rec.get("items") or []:
            user_ids.append(uid)
            item_ids.append(str(it.get("item_id")))
            names.append(it.get("item_name"))
            total.append(it.get("playtime_forever") or 0)
            recent.append(it.get("playtime_2weeks") or 0)

    df = pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "item_name": names,
        "playtime_forever": pd.array(total, dtype="int32"),
        "playtime_2weeks": pd.array(recent, dtype="int32"),
    })
    return df


# ---------------------------------------------------------------------------
# Metadati dei giochi (per il content-based)
# ---------------------------------------------------------------------------

GAME_COLUMNS = [
    "id", "app_name", "title", "developer", "publisher",
    "genres", "tags", "specs", "release_date", "price", "early_access",
]


def load_games(path: Path = GAMES_FILE, limit: Optional[int] = None) -> pd.DataFrame:
    """Una riga per gioco.

    Colonne: item_id (rinominata da 'id', per il join con le librerie),
    app_name, title, developer, publisher, genres, tags, specs, release_date,
    price, early_access. genres, tags e specs sono sempre liste (vuote se mancano).

    Alcuni record non hanno l'id: senza di esso non si possono collegare alle
    librerie, quindi vengono scartati.
    """
    rows = []
    for rec in read_records(path, limit=limit):
        rows.append({col: rec.get(col) for col in GAME_COLUMNS})

    df = pd.DataFrame(rows, columns=GAME_COLUMNS)
    for col in ("genres", "tags", "specs"):
        df[col] = df[col].apply(_as_list)

    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(str)
    df = df.rename(columns={"id": "item_id"})
    df = df.drop_duplicates(subset="item_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Recensioni (voto esplicito: consigliato sì/no)
# ---------------------------------------------------------------------------

def load_reviews(path: Path = REVIEWS_FILE, limit: Optional[int] = None) -> pd.DataFrame:
    """Una riga per recensione.

    Colonne: user_id, item_id, recommend (True = consigliato), posted, review.

    limit: numero massimo di utenti da leggere.
    """
    user_ids, item_ids, recommend, posted, text = [], [], [], [], []

    for rec in read_records(path, limit=limit):
        uid = str(rec.get("user_id"))
        for rv in rec.get("reviews") or []:
            user_ids.append(uid)
            item_ids.append(str(rv.get("item_id")))
            recommend.append(bool(rv.get("recommend")))
            posted.append(rv.get("posted"))
            text.append(rv.get("review"))

    return pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "recommend": recommend,
        "posted": posted,
        "review": text,
    })


# ---------------------------------------------------------------------------
# Controllo veloce: python -m src.load
# ---------------------------------------------------------------------------

def _summary() -> None:
    print("Lettura librerie utenti...")
    ui = load_users_items()
    print(f"  righe (utente, gioco): {len(ui):,}")
    print(f"  utenti con almeno un gioco: {ui['user_id'].nunique():,}")
    print(f"  giochi distinti: {ui['item_id'].nunique():,}")
    print(f"  righe con 0 minuti di gioco: {(ui['playtime_forever'] == 0).mean():.1%}")

    print("\nLettura metadati giochi...")
    games = load_games()
    print(f"  giochi con metadati: {len(games):,}")
    for col in ("genres", "tags", "specs"):
        empty = (games[col].str.len() == 0).mean()
        print(f"  giochi senza {col}: {empty:.1%}")

    played = set(ui["item_id"])
    covered = played & set(games["item_id"])
    print(f"\n  giochi delle librerie che hanno metadati: {len(covered):,} su {len(played):,}")

    print("\nLettura recensioni...")
    rv = load_reviews()
    print(f"  recensioni: {len(rv):,}")
    print(f"  utenti che hanno recensito: {rv['user_id'].nunique():,}")
    print(f"  recensioni positive: {rv['recommend'].mean():.1%}")


if __name__ == "__main__":
    _summary()
