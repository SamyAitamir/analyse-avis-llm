import pandas as pd
import ollama
import time

df = pd.read_csv("data/allocine_sample.csv").sample(120, random_state=42)

def sentiment_llm(texte):
    prompt = f"""Classe le sentiment de cet avis de film.
Réponds uniquement par un seul mot : positif ou négatif.

Avis : {texte}"""
    r = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip().lower()

start = time.time()
df["pred_brut"] = df["review"].apply(sentiment_llm)
print(f"Temps total : {time.time() - start:.1f}s")

df.to_csv("data/resultats_llm.csv", index=False)
print("Résultats sauvegardés dans data/resultats_llm.csv")