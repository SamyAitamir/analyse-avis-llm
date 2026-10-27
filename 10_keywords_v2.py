import pandas as pd
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
stop_fr = stopwords.words("french")
mots_a_exclure = stop_fr + ["film", "films", "cinéma", "cinema", "cinématographique", 
                              "filmique", "filmé", "filmée", "filmées", "dvd"]

df = pd.read_csv("data/allocine_sample.csv")
kw_model = KeyBERT()

negatifs = " ".join(df[df["label"] == 0]["review"].tolist())
positifs = " ".join(df[df["label"] == 1]["review"].tolist())

print("=== Mots-clés des avis NÉGATIFS ===")
for mot, score in kw_model.extract_keywords(negatifs, keyphrase_ngram_range=(1, 2), 
                                               stop_words=mots_a_exclure, top_n=15):
    print(f"{mot}: {score:.3f}")

print("\n=== Mots-clés des avis POSITIFS ===")
for mot, score in kw_model.extract_keywords(positifs, keyphrase_ngram_range=(1, 2), 
                                               stop_words=mots_a_exclure, top_n=15):
    print(f"{mot}: {score:.3f}")