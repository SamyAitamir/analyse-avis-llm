"""Mistral sur les 500 avis de data/allocine_sample.csv, avec les deux prompts
de sentiment (binaire de 02_sentiment_llm.py, échelle 1-5 de 08_sentiment_llm_stars.py),
mot pour mot.

- Sauvegarde incrémentale : une ligne écrite (et flushée) par avis traité.
- Reprenable : relancer le script reprend là où il s'était arrêté (clé = id).
- Seed fixée (42) pour rendre les sorties reproductibles ; la température reste
  celle d'Ollama par défaut, comme dans les scripts 02 et 08.

Sorties : data/resultats_llm_500.csv, data/resultats_llm_stars_500.csv
"""
import csv
import os
import sys
import time

import ollama
import pandas as pd

MODELE = "mistral"
OPTIONS = {"seed": 42}


def prompt_binaire(texte):
    return f"""Classe le sentiment de cet avis de film.
Réponds uniquement par un seul mot : positif ou négatif.

Avis : {texte}"""


def prompt_etoiles(texte):
    return f"""Note cet avis de film sur une échelle de 1 à 5 :
1 = très négatif, 2 = négatif, 3 = neutre/mitigé, 4 = positif, 5 = très positif.
Réponds uniquement par un chiffre entre 1 et 5, sans aucun autre texte.

Avis : {texte}"""


TACHES = [
    ("binaire", prompt_binaire, "data/resultats_llm_500.csv", "pred_brut"),
    ("etoiles", prompt_etoiles, "data/resultats_llm_stars_500.csv", "note_brute"),
]


def verifier_ollama():
    try:
        modeles = [m["model"] for m in ollama.list()["models"]]
    except Exception as e:  # serveur arrêté
        sys.exit(f"Ollama ne répond pas ({e}). Lancer le serveur Ollama puis relancer.")
    if not any(m.startswith(MODELE) for m in modeles):
        sys.exit(f"Modèle {MODELE} absent d'Ollama : {modeles}")
    charges = [m["model"] for m in ollama.ps()["models"]]
    autres = [m for m in charges if not m.startswith(MODELE)]
    if autres:
        sys.exit(f"Autres modèles chargés en VRAM : {autres}. Les décharger d'abord.")
    print("Ollama OK, modèles chargés :", charges or "aucun", flush=True)


def ids_deja_traites(chemin):
    if not os.path.exists(chemin):
        return set()
    try:
        return set(pd.read_csv(chemin, on_bad_lines="skip")["id"].astype(int))
    except (pd.errors.EmptyDataError, KeyError):
        return set()


def main():
    verifier_ollama()
    df = pd.read_csv("data/allocine_sample.csv")
    df["id"] = range(len(df))

    for nom, fprompt, chemin, colonne in TACHES:
        faits = ids_deja_traites(chemin)
        reste = df[~df["id"].isin(faits)]
        print(f"[{nom}] déjà traités : {len(faits)}, restants : {len(reste)}", flush=True)
        nouveau = not os.path.exists(chemin) or os.path.getsize(chemin) == 0
        with open(chemin, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if nouveau:
                w.writerow(["id", "review", "label", colonne, "duree_s"])
            t_debut = time.time()
            for k, row in enumerate(reste.itertuples(index=False), 1):
                t0 = time.perf_counter()
                r = ollama.chat(model=MODELE, options=OPTIONS,
                                messages=[{"role": "user", "content": fprompt(row.review)}])
                rep = r["message"]["content"].strip()
                if nom == "binaire":
                    rep = rep.lower()  # même post-traitement que 02_sentiment_llm.py
                w.writerow([row.id, row.review, row.label, rep, round(time.perf_counter() - t0, 3)])
                f.flush()
                if k % 25 == 0 or k == len(reste):
                    print(f"[{nom}] {k}/{len(reste)} ({time.time() - t_debut:.0f}s)", flush=True)
    print("Terminé.", flush=True)


if __name__ == "__main__":
    main()
