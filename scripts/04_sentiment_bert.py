import pandas as pd
import torch
from transformers import pipeline
import time

device = 0 if torch.cuda.is_available() else -1
print("Device utilisé :", "GPU" if device == 0 else "CPU")

classifier = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=device
)

df = pd.read_csv("data/allocine_sample.csv")  # les 500 avis cette fois, BERT est rapide

def etoiles_vers_label(résultat):
    note = int(résultat["label"][0])  # ex: "4 stars" -> 4
    if note <= 2:
        return 0  # négatif
    elif note >= 4:
        return 1  # positif
    else:
        return None  # 3 étoiles = ambigu, neutre

start = time.time()
resultats = classifier(df["review"].tolist(), truncation=True, batch_size=16)
print(f"Temps total : {time.time() - start:.1f}s")

df["pred"] = [etoiles_vers_label(r) for r in resultats]
df["note_brute"] = [r["label"] for r in resultats]

df.to_csv("data/resultats_bert.csv", index=False)
print("Résultats sauvegardés dans data/resultats_bert.csv")
print(df["pred"].value_counts(dropna=False))