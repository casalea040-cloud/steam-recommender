# Steam Recommender

Progetto per il corso di **Sistemi Intelligenti per Internet** (Università Roma Tre).

Sistema di raccomandazione per videogiochi che confronta un approccio **content-based**
(giochi rappresentati tramite generi, tag e specs) con il **collaborative filtering**
(user-based e item-based), usando i dataset pubblici di Steam della UCSD.
Le preferenze degli utenti sono ricavate dalle ore di gioco, corrette con le
recensioni quando presenti.

## Struttura

```
steam-recommender/
├── data/
│   ├── raw/            # file scaricati (non caricati su GitHub)
│   ├── interim/        # gli stessi dati letti e salvati come tabelle Parquet
│   └── processed/      # tabelle pulite e divisione training/test
├── notebooks/          # esplorazione dei dati
├── scripts/
│   └── download_data.py
├── src/
│   ├── config.py       # percorsi e parametri del progetto
│   ├── load.py         # lettura dei file grezzi in tabelle pandas
│   ├── preprocess.py   # pulizia dei dati e indice di gradimento
│   ├── split.py        # divisione training/test
│   ├── matrices.py     # matrici utenti x giochi (gradimento e possesso)
│   ├── collaborative.py # user-based CF e baseline di popolarità
│   └── evaluate.py     # confronto dei modelli con precision e recall
├── report/             # report del progetto (LaTeX)
├── app/                # web app Streamlit (fasi successive)
├── requirements.txt
└── README.md
```

## Installazione

Dalla cartella del progetto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dati

```bash
python scripts/download_data.py
```

Lo script scarica in `data/raw/`:

| File | Contenuto | Uso |
|---|---|---|
| `australian_users_items.json.gz` | librerie degli utenti con le ore di gioco | collaborative filtering |
| `steam_games.json.gz` | metadati dei giochi: generi, tag, specs, prezzo... | content-based |
| `australian_user_reviews.json.gz` | recensioni con voto consigliato sì/no | indice di gradimento |

Se il download automatico non riesce, i file si scaricano a mano dalla pagina
[Recommender Systems and Personalization Datasets](https://cseweb.ucsd.edu/~jmcauley/datasets.html)
(sezione *Steam Video Game and Bundle Data*) e vanno messi in `data/raw/`.

La prima volta che si lancia `src.preprocess`, i file vengono letti e salvati come
tabelle Parquet in `data/interim/` (qualche minuto); le volte successive si
rileggono da lì in pochi secondi.

## Pulizia, divisione e valutazione

Dalla cartella del progetto, con l'ambiente virtuale attivo:

```bash
python -m src.preprocess   # pulizia e indice di gradimento → data/processed/
python -m src.split        # divisione training/test → data/processed/test.parquet
python -m src.evaluate     # confronto tra popolarità e user-based CF
```

`src.split` è facoltativo: se `test.parquet` non esiste, `src.evaluate` lo crea da sé.
La valutazione accetta alcune opzioni:

```bash
python -m src.evaluate --n 5          # numero di giochi raccomandati (default 10)
python -m src.evaluate --k 20         # numero di vicini dello user-based (default 50)
python -m src.evaluate --sample 5000  # valuta solo un campione di utenti, più veloce
```

Le soglie della pulizia, la quota del test e il seme casuale sono in `src/config.py`.

## Fonte dei dati

Dataset di Julian McAuley (UC San Diego). Articoli da citare:

- W.-C. Kang, J. McAuley. *Self-attentive sequential recommendation*. ICDM, 2018.
- M. Wan, J. McAuley. *Item recommendation on monotonic behavior chains*. RecSys, 2018.
- A. Pathak, K. Gupta, J. McAuley. *Generating and personalizing bundle recommendations on Steam*. SIGIR, 2017.
