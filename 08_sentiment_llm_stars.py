import pandas as pd
import ollama
import re
import time

df = pd.read_csv("data/resultats_llm.csv")  # on reprend les mêmes 120 avis

def note_llm(texte):
    prompt = f"""Note cet avis de film sur une échelle de 1 à 5 :
1 = très négatif, 2 = négatif, 3 = neutre/mitigé, 4 = positif, 5 = très positif.
Réponds uniquement par un chiffre entre 1 et 5, sans aucun autre texte.

Avis : {texte}"""
    r = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip()

start = time.time()
df["note_brute"] = df["review"].apply(note_llm)
print(f"Temps total : {time.time() - start:.1f}s")

df.to_csv("data/resultats_llm_stars.csv", index=False)
print(df["note_brute"].value_counts())