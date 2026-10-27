import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("data/resultats_bert.csv")

non_exploitables = df["pred"].isna().sum()
print(f"Avis neutres (3 étoiles), exclus de l'évaluation : {non_exploitables}/{len(df)}")

df_valide = df.dropna(subset=["pred"])
print(f"\nAccuracy : {accuracy_score(df_valide['label'], df_valide['pred']):.3f}")
print("\nMatrice de confusion :")
print(confusion_matrix(df_valide["label"], df_valide["pred"]))
print("\nRapport détaillé :")
print(classification_report(df_valide["label"], df_valide["pred"], target_names=["négatif", "positif"]))