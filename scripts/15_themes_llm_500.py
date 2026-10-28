"""Catégorisation thématique par Mistral sur les 500 avis de data/allocine_sample.csv
(tirage stratifié 250/250 reproductible de 01_dataset.py, random_state=42).

Prompt et catégories identiques à 12_themes_llm.py (ex-11_themes_llm.py), qui ne
traitait que 30 avis et écrit dans data/resultats_themes_llm.csv : ce script ne le
relance pas pour ne pas écraser ce fichier.

- Sauvegarde incrémentale et reprenable (clé = id).
- Seed fixée (42), température par défaut d'Ollama.
- La réponse brute est conservée (minuscules, strip, comme 12_themes_llm.py) ;
  la normalisation est faite dans 17_analyse_themes.py.

Sortie : data/resultats_themes_llm_500.csv
"""
import csv
import os
import sys
import time

import ollama
import pandas as pd

MODELE = "mistral"
OPTIONS = {"seed": 42}
SORTIE = "data/resultats_themes_llm_500.csv"

categories = ["scénario", "jeu d'acteur", "réalisation/mise en scène", "rythme", "humour", "émotion", "autre"]


def prompt_theme(texte):
    return f"""Voici un avis de film. Identifie la catégorie principale sur laquelle porte la critique, parmi : {", ".join(categories)}.
Réponds uniquement par le nom de la catégorie, sans explication.

Avis : {texte}"""


def verifier_ollama():
    try:
        modeles = [m["model"] for m in ollama.list()["models"]]
    except Exception as e:
        sys.exit(f"Ollama ne répond pas ({e}). Lancer le serveur Ollama puis relancer.")
    if not any(m.startswith(MODELE) for m in modeles):
        sys.exit(f"Modèle {MODELE} absent d'Ollama : {modeles}")
    autres = [m["model"] for m in ollama.ps()["models"] if not m["model"].startswith(MODELE)]
    if autres:
        sys.exit(f"Autres modèles chargés en VRAM : {autres}. Les décharger d'abord.")
    print("Ollama OK", flush=True)


def main():
    verifier_ollama()
    df = pd.read_csv("data/allocine_sample.csv")
    df["id"] = range(len(df))

    faits = set()
    if os.path.exists(SORTIE) and os.path.getsize(SORTIE) > 0:
        faits = set(pd.read_csv(SORTIE, on_bad_lines="skip")["id"].astype(int))
    reste = df[~df["id"].isin(faits)]
    print(f"[themes] déjà traités : {len(faits)}, restants : {len(reste)}", flush=True)

    nouveau = not faits and not (os.path.exists(SORTIE) and os.path.getsize(SORTIE) > 0)
    with open(SORTIE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nouveau:
            w.writerow(["id", "review", "label", "theme", "duree_s"])
        t_debut = time.time()
        for k, row in enumerate(reste.itertuples(index=False), 1):
            t0 = time.perf_counter()
            r = ollama.chat(model=MODELE, options=OPTIONS,
                            messages=[{"role": "user", "content": prompt_theme(row.review)}])
            w.writerow([row.id, row.review, row.label, r["message"]["content"].strip().lower(),
                        round(time.perf_counter() - t0, 3)])
            f.flush()
            if k % 25 == 0 or k == len(reste):
                print(f"[themes] {k}/{len(reste)} ({time.time() - t_debut:.0f}s)", flush=True)
    print("Terminé.", flush=True)


if __name__ == "__main__":
    main()
