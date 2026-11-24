"""Pulizia dei dati e indice di gradimento.

Parte dalle tabelle di data/interim (src/load.py) e produce in data/processed:
    interactions.parquet  una riga per coppia (utente, gioco) con le ore e il gradimento
    games.parquet         i metadati dei soli giochi rimasti

Passaggi:
    1. elimino le coppie utente-gioco duplicate;
    2. tengo solo i giochi che hanno metadati (il content-based deve poterli descrivere);
    3. collego a ogni coppia l'eventuale recensione dell'utente su quel gioco
       (le recensioni di giochi assenti dalla libreria vengono ignorate);
    4. calcolo l'indice di gradimento: la recensione prevale; senza recensione,
       il gioco è gradito se giocato almeno MIN_PLAYTIME_MINUTES;
    5. tolgo utenti con pochi giochi graditi e giochi graditi da pochi utenti,
       ripetendo finché i conteggi non cambiano più.

Uso, dalla cartella principale del progetto:
    python -m src.preprocess
"""

import numpy as np
import pandas as pd

from src.config import (
    CLEAN_GAMES_TABLE,
    INTERACTIONS_TABLE,
    MIN_LIKED_PER_USER,
    MIN_LIKERS_PER_GAME,
    MIN_PLAYTIME_MINUTES,
    PROCESSED_DIR,
)
from src.load import load_tables


def liking_index(playtime: pd.Series, recommend: pd.Series) -> pd.Series:
    """Indice di gradimento (1 = gradito, 0 = no).

    recommend: True/False se c'è una recensione, mancante (NaN) altrimenti.
    """
    has_review = recommend.notna()
    by_review = recommend.fillna(False).astype(bool)
    by_hours = playtime >= MIN_PLAYTIME_MINUTES
    return pd.Series(np.where(has_review, by_review, by_hours), index=playtime.index).astype("int8")


def filter_users_and_games(df: pd.DataFrame) -> pd.DataFrame:
    """Toglie utenti con meno di MIN_LIKED_PER_USER giochi graditi e giochi con meno
    di MIN_LIKERS_PER_GAME utenti che li gradiscono.

    I due filtri si influenzano: togliere un gioco abbassa il conteggio dei suoi
    utenti, e viceversa. Per questo si ripetono finché non tolgono più niente.
    """
    step = 0
    while True:
        step += 1
        liked_per_user = df.groupby("user_id")["liked"].transform("sum")
        df = df[liked_per_user >= MIN_LIKED_PER_USER]
        likers_per_game = df.groupby("item_id")["liked"].transform("sum")
        kept = df[likers_per_game >= MIN_LIKERS_PER_GAME]

        print(f"  passaggio {step}: {kept['user_id'].nunique():,} utenti, "
              f"{kept['item_id'].nunique():,} giochi, {len(kept):,} coppie")
        if len(kept) == len(df):
            return kept
        df = kept


def preprocess() -> pd.DataFrame:
    users_items, games, reviews = load_tables()

    print("\nPulizia")
    df = users_items.drop_duplicates(subset=["user_id", "item_id"])
    print(f"  coppie utente-gioco duplicate eliminate: {len(users_items) - len(df):,}")

#Tiene solo i giochi con metadati
    df = df[df["item_id"].isin(set(games["item_id"]))]
    print(f"  coppie con giochi dotati di metadati: {len(df):,}")

    reviews = reviews.drop_duplicates(subset=["user_id", "item_id"])
    df = df.merge(reviews[["user_id", "item_id", "recommend"]],
                  on=["user_id", "item_id"], how="left")
    print(f"  recensioni collegate a un gioco in libreria: "
          f"{df['recommend'].notna().sum():,} su {len(reviews):,}")

    df["liked"] = liking_index(df["playtime_forever"], df["recommend"])
    print(f"  coppie con gradimento 1: {df['liked'].mean():.1%}")

    print(f"\nFiltri (almeno {MIN_LIKED_PER_USER} giochi graditi per utente, "
          f"almeno {MIN_LIKERS_PER_GAME} utenti per gioco)")
    df = filter_users_and_games(df)

    df = df[["user_id", "item_id", "playtime_forever", "recommend", "liked"]].reset_index(drop=True)
    clean_games = games[games["item_id"].isin(set(df["item_id"]))].reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERACTIONS_TABLE, index=False)
    clean_games.to_parquet(CLEAN_GAMES_TABLE, index=False)

    print("\nRisultato")
    print(f"  utenti: {df['user_id'].nunique():,}")
    print(f"  giochi: {df['item_id'].nunique():,}")
    print(f"  coppie utente-gioco: {len(df):,}, di cui gradite: {df['liked'].sum():,}")
    print(f"  salvato in {INTERACTIONS_TABLE} e {CLEAN_GAMES_TABLE}")
    return df


if __name__ == "__main__":
    preprocess()
