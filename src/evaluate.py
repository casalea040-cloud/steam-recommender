"""Valutazione offline: precision@n e recall@n.

Per ogni utente il modello raccomanda n giochi usando solo il training; poi si
contano quanti di questi sono tra i giochi graditi nascosti nel test:
    precision@n = giochi indovinati / n
    recall@n    = giochi indovinati / giochi dell'utente nel test
e si fa la media su tutti gli utenti.

Uso, dalla cartella principale del progetto:
    python -m src.evaluate                  # tutti gli utenti, n = 10, k = 50
    python -m src.evaluate --sample 5000    # un campione casuale di utenti, più veloce
    python -m src.evaluate --k 20 --n 20
"""

import argparse
import time

import numpy as np
import pandas as pd

from src.collaborative import Popularity, UserBasedCF
from src.config import INTERACTIONS_TABLE, RANDOM_SEED
from src.matrices import Matrices, build_matrices
from src.split import load_split


# Calcola precision@n e recall@n medie di un modello.
#
# Riceve:
#   model   il modello  
#   m       le matrici del training, servono per tradurre righe e colonne
#           in utenti e giochi (m.user_ids, m.item_ids, m.user_index)
#   hidden  per ogni utente da valutare, l'insieme dei suoi giochi nascosti
#           nel test (es. Anna -> {"C"})
#   n       quanti giochi far raccomandare a ogni utente
#
# Come funziona:
#   1. trasforma gli utenti da valutare nei loro numeri di riga nelle matrici;
#   2. chiede al modello gli n giochi raccomandati a tutti questi utenti
#      (recommend_all): recs ha una riga per utente, nello stesso ordine di
#      hidden, e in ogni riga i numeri di colonna degli n giochi consigliati;
#   3. per l'i-esimo utente traduce i numeri di colonna raccomandati in item_id
#      e conta quanti sono tra i suoi giochi nascosti (intersezione tra insiemi):
#      sono i giochi "indovinati";
#   4. precision = indovinati / n, recall = indovinati / giochi nascosti
#      dell'utente;
#   5. restituisce la media di precision e recall su tutti gli utenti.
def evaluate(model, m: Matrices, hidden: pd.Series, n: int) -> dict:
    rows = np.array([m.user_index[u] for u in hidden.index])
    recs = model.recommend_all(rows, n)

    precision, recall = [], []
    for i in range(len(rows)):
        truth = hidden.iloc[i]                       # giochi nascosti dell'i-esimo utente
        recommended = set(m.item_ids[recs[i]])       # i suoi giochi consigliati, come item_id
        hits = len(recommended & truth) #Simbolo di intersezione &, prende quindi effettivamente
        precision.append(hits / n)      #il numero di giochi presenti sia in reccomended che in truth(ovvero quelli correttamente raccomandati dal modello) 
        recall.append(hits / len(truth))
    return {f"precision@{n}": np.mean(precision), f"recall@{n}": np.mean(recall)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta user-based CF e popolarità.")
    #Argomenti passabili quando viene chiamato il file evaluate.py:
    parser.add_argument("--n", type=int, default=10, help="giochi raccomandati per utente")
    parser.add_argument("--k", type=int, default=50, help="vicini dello user-based")
    parser.add_argument("--sample", type=int, default=0,
                        help="numero di utenti da valutare (0 = tutti)")
    args = parser.parse_args()

    df = pd.read_parquet(INTERACTIONS_TABLE)
    train, test = load_split()
    m = build_matrices(train, df)

    # Per ogni utente, l'insieme dei giochi nascosti nel test
    hidden = test.groupby("user_id")["item_id"].apply(set)
    if args.sample and args.sample < len(hidden):
        hidden = hidden.sample(args.sample, random_state=RANDOM_SEED)
    print(f"Utenti valutati: {len(hidden):,} - giochi raccomandati: {args.n}\n")

    models = {
        "Popolarità": Popularity(),
        f"User-based CF (k = {args.k})": UserBasedCF(k=args.k),
    }
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(m)
        results[name] = evaluate(model, m, hidden, args.n)
        print(f"{name}: fatto in {time.time() - start:.0f} s")

    print()
    print(pd.DataFrame(results).T.round(4).to_string())


if __name__ == "__main__":
    main()
