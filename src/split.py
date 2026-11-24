"""Divisione dei dati in training e test.

Per ogni utente nascondo una quota (TEST_FRACTION) dei giochi che ha gradito:
quei giochi formano il test, e i modelli non devono vederli. Tutto il resto
(compresi i giochi posseduti ma non graditi) resta nel training.

Ogni utente ha almeno MIN_LIKED_PER_USER giochi graditi (vedi src/preprocess.py),
quindi gliene restano sempre alcuni nel training e almeno uno va nel test.

Uso, dalla cartella principale del progetto:
    python -m src.split
"""

from typing import Tuple

import pandas as pd

from src.config import INTERACTIONS_TABLE, RANDOM_SEED, TEST_FRACTION, TEST_TABLE


def train_test_split(df: pd.DataFrame, fraction: float = TEST_FRACTION,
                     seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Restituisce (training, test).

    Per ogni utente estraggo a caso una quota `fraction` dei giochi che ha gradito:
    quelli vanno nel test, tutto il resto resta nel training.
    """
    liked = df[df["liked"] == 1]
    test = liked.groupby("user_id").sample(frac=fraction, random_state=seed)
    train = df.drop(test.index)
    return train.reset_index(drop=True), test[["user_id", "item_id"]].reset_index(drop=True)


def remove_test(df: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Le righe di df che non compaiono nel test."""
    merged = df.merge(test, on=["user_id", "item_id"], how="left", indicator=True)
    return merged[merged["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)


def load_split() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """La divisione salvata su disco; se non c'è ancora, la crea e la salva."""
    df = pd.read_parquet(INTERACTIONS_TABLE)
    if TEST_TABLE.exists():
        test = pd.read_parquet(TEST_TABLE)
        return remove_test(df, test), test

    train, test = train_test_split(df)
    test.to_parquet(TEST_TABLE, index=False)
    return train, test


if __name__ == "__main__":
    df = pd.read_parquet(INTERACTIONS_TABLE)
    train, test = train_test_split(df)
    test.to_parquet(TEST_TABLE, index=False)

    print(f"Coppie nel training: {len(train):,}")
    print(f"Giochi graditi nascosti per il test: {len(test):,} "
          f"({len(test) / df['liked'].sum():.1%} dei giochi graditi)")
    per_user = test.groupby("user_id").size()
    print(f"Giochi nel test per utente: minimo {per_user.min()}, "
          f"mediana {per_user.median():.0f}, massimo {per_user.max()}")
    print(f"Salvato in {TEST_TABLE}")
