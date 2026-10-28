import pandas as pd

df = pd.read_csv("data/resultats_llm.csv")
df["pred"] = df["pred_brut"].apply(lambda p: 1 if "positif" in p.lower() else 0)

erreurs = df[df["label"] != df["pred"]]
for _, row in erreurs.iterrows():
    print(f"Vrai: {row['label']} | Prédit: {row['pred']}")
    print(row["review"][:200])
    print("---")