from datasets import load_dataset
import os

ds = load_dataset("tblard/allocine")
df = ds["test"].to_pandas()     # 20 000 avis

print(df.shape)
print(df["label"].value_counts())   # 0 = négatif, 1 = positif
print(df["review"].str.len().describe())

# Nettoyage minimal
df = df.drop_duplicates(subset="review")
df = df[df["review"].str.strip().str.len() > 20]

# Échantillon équilibré de 500 avis (250 positifs / 250 négatifs)
sample = (df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(250, random_state=42)))

os.makedirs("data", exist_ok=True)
sample.to_csv("data/allocine_sample.csv", index=False)
print("Échantillon sauvegardé :", sample.shape)