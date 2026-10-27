import pandas as pd
import ollama

df = pd.read_csv("data/allocine_sample.csv").sample(30, random_state=42)

categories = ["scénario", "jeu d'acteur", "réalisation/mise en scène", "rythme", "humour", "émotion", "autre"]

def theme_llm(texte):
    prompt = f"""Voici un avis de film. Identifie la catégorie principale sur laquelle porte la critique, parmi : {", ".join(categories)}.
Réponds uniquement par le nom de la catégorie, sans explication.

Avis : {texte}"""
    r = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip().lower()

df["theme"] = df["review"].apply(theme_llm)
df.to_csv("data/resultats_themes_llm.csv", index=False)
print(df["theme"].value_counts())
print(df[["review", "theme"]].head(10))