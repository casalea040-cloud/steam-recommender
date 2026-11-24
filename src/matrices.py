"""Le matrici utenti x giochi usate dal collaborative filtering.

Dalla tabella del training (una riga per coppia utente-gioco) costruisco due
matrici sparse, con gli utenti sulle righe e i giochi sulle colonne:

    G (gradimento): 1 se l'utente ha gradito il gioco, 0 altrimenti
    O (possesso):   1 se l'utente possiede il gioco, 0 altrimenti

G serve a calcolare similarità e punteggi; O serve solo a escludere dalle
raccomandazioni i giochi che l'utente ha già. Le matrici sono sparse: si salvano
solo gli 1, e tutte le altre celle valgono 0.

Righe e colonne sono numerate da 0; user_ids[r] e item_ids[c] dicono a quale
utente e a quale gioco corrispondono la riga r e la colonna c.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass
class Matrices:
    G: sparse.csr_matrix          # gradimento (utenti x giochi)
    O: sparse.csr_matrix          # possesso (utenti x giochi)
    user_ids: np.ndarray          # user_ids[r] = utente della riga r
    item_ids: np.ndarray          # item_ids[c] = gioco della colonna c
    user_index: Dict[str, int]    # utente -> riga
    item_index: Dict[str, int]    # gioco -> colonna


def build_matrices(train: pd.DataFrame, all_rows: pd.DataFrame) -> Matrices:
    """Costruisce G e O dal training.

    all_rows: tutte le coppie (training + test); serve solo a fissare l'elenco
    completo di utenti e giochi, così i giochi del test hanno comunque la loro
    colonna e possono essere raccomandati.
    """
    user_ids = np.sort(all_rows["user_id"].unique())
    item_ids = np.sort(all_rows["item_id"].unique())
    user_index = {u: r for r, u in enumerate(user_ids)}
    item_index = {i: c for c, i in enumerate(item_ids)}

    rows = train["user_id"].map(user_index).to_numpy()
    cols = train["item_id"].map(item_index).to_numpy()
    shape = (len(user_ids), len(item_ids))

    owned = np.ones(len(train), dtype=np.float32)
    O = sparse.csr_matrix((owned, (rows, cols)), shape=shape)

    liked = train["liked"].to_numpy() == 1
    G = sparse.csr_matrix((np.ones(liked.sum(), dtype=np.float32),
                           (rows[liked], cols[liked])), shape=shape)

    return Matrices(G, O, user_ids, item_ids, user_index, item_index)
