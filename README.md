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
│   └── processed/      # tabelle pulite (fasi successive)
├── notebooks/          # esplorazione dei dati
├── scripts/
│   └── download_data.py
├── src/
│   ├── config.py       # percorsi del progetto
│   └── load.py         # lettura dei file grezzi in tabelle pandas
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

Controllo rapido del contenuto:

```bash
python -m src.load
```

## Fonte dei dati

Dataset di Julian McAuley (UC San Diego). Articoli da citare:

- W.-C. Kang, J. McAuley. *Self-attentive sequential recommendation*. ICDM, 2018.
- M. Wan, J. McAuley. *Item recommendation on monotonic behavior chains*. RecSys, 2018.
- A. Pathak, K. Gupta, J. McAuley. *Generating and personalizing bundle recommendations on Steam*. SIGIR, 2017.
