import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/resultats_llm.csv")

def normaliser(pred):
    pred = pred.lower()
    if "positif" in pred:
        return 1
    elif "négatif" in pred or "negatif" in pred:
        return 0
    else:
        return None  # réponse non exploitable

df["pred"] = df["pred_brut"].apply(normaliser)

non_exploitables = df["pred"].isna().sum()
print(f"Réponses non exploitables : {non_exploitables}/{len(df)}")

df_valide = df.dropna(subset=["pred"])
print(f"\nAccuracy : {accuracy_score(df_valide['label'], df_valide['pred']):.3f}")
print("\nMatrice de confusion :")
print(confusion_matrix(df_valide["label"], df_valide["pred"]))
print("\nRapport détaillé :")
print(classification_report(df_valide["label"], df_valide["pred"], target_names=["négatif", "positif"]))