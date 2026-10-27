"""Mesure du temps d'inférence de BERT nlptown sur les 500 avis, avec les
mêmes paramètres que 04_sentiment_bert.py, et vérification que les notes
obtenues sont identiques à celles de data/resultats_bert.csv (déterminisme).
"""
import time
import pandas as pd
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("sentiment-analysis",
               model="nlptown/bert-base-multilingual-uncased-sentiment", device=device)
df = pd.read_csv("data/resultats_bert.csv")
clf(df["review"].tolist()[:16], truncation=True, batch_size=16)  # préchauffage
t0 = time.perf_counter()
res = clf(df["review"].tolist(), truncation=True, batch_size=16)
duree = time.perf_counter() - t0
identiques = sum(r["label"] == n for r, n in zip(res, df["note_brute"]))
print(f"device={'GPU' if device == 0 else 'CPU'} total={duree:.1f}s "
      f"par_avis={duree/len(df)*1000:.1f}ms identiques={identiques}/{len(df)}")
