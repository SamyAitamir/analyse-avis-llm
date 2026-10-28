import pandas as pd
import re
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("data/resultats_llm_stars.csv")

def parse_note(s):
    s = str(s)
    m = re.search(r"[1-5]", s)
    return int(m.group()) if m else None

df["note"] = df["note_brute"].apply(parse_note)

def binariser(note):
    if note in [1, 2]:
        return 0
    elif note in [4, 5]:
        return 1
    else:
        return None  # 3, ou réponse non parsable

df["pred"] = df["note"].apply(binariser)

exclus = df["pred"].isna().sum()
print(f"Avis exclus (neutres ou non parsables) : {exclus}/{len(df)}")

df_valide = df.dropna(subset=["pred"])
print(f"\nAccuracy (même principe que BERT) : {accuracy_score(df_valide['label'], df_valide['pred']):.3f}")
print(classification_report(df_valide["label"], df_valide["pred"], target_names=["négatif", "positif"]))