import pandas as pd
from keybert import KeyBERT
from collections import Counter
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_fr = stopwords.words("french")
mots_a_exclure = stop_fr + ["film", "films", "cinéma", "cinema", "cinématographique",
                              "filmique", "filmé", "filmée", "filmées", "dvd"]

df = pd.read_csv("data/allocine_sample.csv")
kw_model = KeyBERT()

def extraire_mots(textes, top_n=5):
    compteur = Counter()
    for texte in textes:
        mots = kw_model.extract_keywords(texte, keyphrase_ngram_range=(1, 2),
                                           stop_words=mots_a_exclure, top_n=top_n)
        for mot, score in mots:
            compteur[mot] += 1
    return compteur

negatifs = df[df["label"] == 0]["review"].tolist()
positifs = df[df["label"] == 1]["review"].tolist()

compteur_neg = extraire_mots(negatifs)
compteur_pos = extraire_mots(positifs)

print("=== Thèmes les plus fréquents — NÉGATIFS ===")
for mot, count in compteur_neg.most_common(15):
    print(f"{mot}: {count}")

print("\n=== Thèmes les plus fréquents — POSITIFS ===")
for mot, count in compteur_pos.most_common(15):
    print(f"{mot}: {count}")