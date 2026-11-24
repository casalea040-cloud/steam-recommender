"""Collaborative filtering user-based.

Per raccomandare all'utente A:
    1. similarità coseno tra la riga di A in G e le righe di tutti gli altri utenti;
    2. i k utenti più simili ad A sono i suoi vicini;
    3. per ogni gioco g, punteggio = somma delle similarità dei vicini che hanno
       gradito g, divisa per la somma delle similarità di tutti i vicini
       (frazione pesata dei vicini che hanno gradito g, tra 0 e 1);
    4. i giochi che A possiede già (matrice O) vengono esclusi;
    5. si prendono gli n giochi con il punteggio più alto.

La matrice delle similarità utenti x utenti non viene mai costruita intera
(sarebbe troppo grande): gli utenti vengono elaborati a blocchi, e di ogni
blocco si tengono solo le raccomandazioni.
"""

from typing import Tuple

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from src.matrices import Matrices

# Contributo minimo della popolarità al punteggio: non cambia l'ordine tra giochi
# con punteggi diversi, ma scioglie i pareggi (per esempio tra i tanti giochi a 0)
# a favore dei giochi più graditi in generale
TIE_BREAK = 1e-6


class UserBasedCF:
    def __init__(self, k: int = 50):
        self.k = k  # numero di vicini

    def fit(self, m: Matrices) -> "UserBasedCF":
        self.G = m.G.tocsr()
        self.O = m.O.tocsr()
        # Ogni riga divisa per la sua norma: il coseno diventa un prodotto scalare
        self.Gn = normalize(self.G, norm="l2", axis=1).tocsr()
        likers = np.asarray(self.G.sum(axis=0)).ravel()
        self.popularity = likers / likers.max()
        return self

    def neighbors(self, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """I k vicini di ogni utente in `rows`: (indici dei vicini, similarità).

        Restituisce due matrici len(rows) x k.
        """
        # Similarità coseno di ogni utente del blocco con tutti gli utenti
        block = self.Gn[rows].toarray().T                # giochi x blocco
        sims = np.asarray(self.Gn @ block).T              # blocco x utenti
        sims[np.arange(len(rows)), rows] = 0              # un utente non è vicino di se stesso

        k = min(self.k, sims.shape[1] - 1)
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        weights = np.take_along_axis(sims, idx, axis=1)
        return idx, weights

    def scores(self, rows: np.ndarray) -> np.ndarray:
        """Punteggio di ogni gioco per ogni utente in `rows` (len(rows) x giochi)."""
        idx, weights = self.neighbors(rows)

        # Matrice sparsa blocco x utenti con le similarità dei soli k vicini:
        # moltiplicata per G somma, per ogni gioco, le similarità dei vicini che l'hanno gradito
        n_rows, k = idx.shape
        W = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(n_rows), k), idx.ravel())),
                              shape=(n_rows, self.G.shape[0]))
        numerator = (W @ self.G).toarray()
        denominator = weights.sum(axis=1, keepdims=True)
        # Se un utente non ha nessun vicino con similarità > 0 il denominatore è 0:
        # i suoi punteggi restano 0 e decide la popolarità (TIE_BREAK)
        return np.divide(numerator, denominator, out=np.zeros_like(numerator),
                         where=denominator > 0)

    def recommend(self, rows: np.ndarray, n: int = 10) -> np.ndarray:
        """Gli n giochi raccomandati (indici di colonna) per ogni utente in `rows`,
        dal più consigliato al meno consigliato."""
        s = self.scores(rows) + TIE_BREAK * self.popularity
        s[self.O[rows].toarray() > 0] = -np.inf          # esclude i giochi già posseduti
        top = np.argpartition(-s, n - 1, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)

    def recommend_all(self, rows: np.ndarray, n: int = 10, block_size: int = 500) -> np.ndarray:
        """Gli n giochi raccomandati a ogni utente in `rows`: una riga per utente,
        nello stesso ordine di `rows`.

        I calcoli si fanno block_size utenti alla volta, perché le similarità di
        tutti gli utenti con tutti gli altri non starebbero in memoria; i risultati
        dei blocchi (pochi numeri per utente) vengono poi messi uno sotto l'altro tramite vstack
        """
        results = []
        for start in range(0, len(rows), block_size):
            block = rows[start:start + block_size]
            results.append(self.recommend(block, n))
        return np.vstack(results)


class Popularity:
    """Baseline: a tutti gli utenti gli stessi giochi, i più graditi nel training,
    esclusi quelli che l'utente possiede già."""

    def fit(self, m: Matrices) -> "Popularity":
        self.O = m.O.tocsr()
        self.likers = np.asarray(m.G.sum(axis=0)).ravel().astype(float)
        return self

    def recommend(self, rows: np.ndarray, n: int = 10) -> np.ndarray:
        s = np.tile(self.likers, (len(rows), 1))
        s[self.O[rows].toarray() > 0] = -np.inf
        top = np.argpartition(-s, n - 1, axis=1)[:, :n]
        order = np.argsort(-np.take_along_axis(s, top, axis=1), axis=1)
        return np.take_along_axis(top, order, axis=1)

    def recommend_all(self, rows: np.ndarray, n: int = 10, block_size: int = 500) -> np.ndarray:
        """Come UserBasedCF.recommend_all: una riga di n giochi per ogni utente."""
        results = []
        for start in range(0, len(rows), block_size):
            block = rows[start:start + block_size]
            results.append(self.recommend(block, n))
        return np.vstack(results)
