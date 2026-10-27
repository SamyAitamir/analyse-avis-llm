"""Reproduit exactement 11_keywords_v3.py (extraction par avis, top 5, agrégation
par fréquence) et sauvegarde la distribution complète des expressions extraites,
pour quantifier la dispersion des résultats (le script d'origine n'affiche que
le top 15)."""
import json
from collections import Counter
import nltk
import pandas as pd
from keybert import KeyBERT
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
mots_a_exclure = stopwords.words("french") + ["film", "films", "cinéma", "cinema", "cinématographique",
                                              "filmique", "filmé", "filmée", "filmées", "dvd"]
df = pd.read_csv("data/allocine_sample.csv")
kw_model = KeyBERT()

stats = {"n_stop_words_nltk": len(stopwords.words("french")), "n_mots_domaine_ajoutes": 10}
for lab, nom in [(0, "negatifs"), (1, "positifs")]:
    textes = df[df["label"] == lab]["review"].tolist()
    c, n_vides = Counter(), 0
    for texte in textes:
        mots = kw_model.extract_keywords(texte, keyphrase_ngram_range=(1, 2),
                                         stop_words=mots_a_exclure, top_n=5)
        n_vides += len(mots) == 0
        for mot, _ in mots:
            c[mot] += 1
    pd.DataFrame(c.most_common(), columns=["terme", "occurrences"]).to_csv(
        f"mesures/resultats/keybert_v3_distribution_{nom}.csv", index=False)
    total = sum(c.values())
    stats[nom] = {
        "n_avis": len(textes), "extractions": total, "termes_distincts": len(c),
        "termes_vus_une_fois": sum(v == 1 for v in c.values()),
        "part_termes_vus_une_fois": round(sum(v == 1 for v in c.values()) / len(c), 3),
        "occurrences_max": c.most_common(1)[0][1],
        "termes_a_2_occurrences_ou_plus": sum(v >= 2 for v in c.values()),
        "part_extractions_top15": round(sum(v for _, v in c.most_common(15)) / total, 3),
        "avis_sans_extraction": n_vides,
    }
with open("mesures/resultats/keybert_v3_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
