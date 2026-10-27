"""Second passage de Mistral avec des prompts STRICTEMENT identiques à ceux de
02_sentiment_llm.py, 08_sentiment_llm_stars.py et 12_themes_llm.py, sur les
mêmes avis, pour mesurer (1) la stabilité d'un run à l'autre (Ollama utilise
par défaut un échantillonnage non déterministe) et (2) les temps d'exécution.
Les fichiers d'origine de data/ ne sont pas modifiés.
"""
import time
import pandas as pd
import ollama

base = pd.read_csv("data/resultats_llm.csv")[["review", "label"]]  # les 120 avis
themes_base = pd.read_csv("data/resultats_themes_llm.csv")[["review", "label"]]  # les 30 avis

categories = ["scénario", "jeu d'acteur", "réalisation/mise en scène", "rythme", "humour", "émotion", "autre"]

def prompt_binaire(texte):
    return f"""Classe le sentiment de cet avis de film.
Réponds uniquement par un seul mot : positif ou négatif.

Avis : {texte}"""

def prompt_etoiles(texte):
    return f"""Note cet avis de film sur une échelle de 1 à 5 :
1 = très négatif, 2 = négatif, 3 = neutre/mitigé, 4 = positif, 5 = très positif.
Réponds uniquement par un chiffre entre 1 et 5, sans aucun autre texte.

Avis : {texte}"""

def prompt_theme(texte):
    return f"""Voici un avis de film. Identifie la catégorie principale sur laquelle porte la critique, parmi : {", ".join(categories)}.
Réponds uniquement par le nom de la catégorie, sans explication.

Avis : {texte}"""

def appel(prompt):
    t0 = time.perf_counter()
    r = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip(), time.perf_counter() - t0

# Préchauffage (chargement du modèle en VRAM), exclu des mesures
appel("Bonjour")

for nom, df, fprompt, post in [
    ("binaire", base.copy(), prompt_binaire, str.lower),
    ("etoiles", base.copy(), prompt_etoiles, lambda s: s),
    ("themes", themes_base.copy(), prompt_theme, str.lower),
]:
    sorties, durees = [], []
    for texte in df["review"]:
        s, d = appel(fprompt(texte))
        sorties.append(post(s))
        durees.append(d)
    df["sortie_rerun"] = sorties
    df["duree_s"] = durees
    df.to_csv(f"mesures/resultats/rerun_mistral_{nom}.csv", index=False)
    print(f"{nom}: {len(df)} avis, total {sum(durees):.1f}s, "
          f"moyenne {sum(durees)/len(durees):.2f}s/avis")
