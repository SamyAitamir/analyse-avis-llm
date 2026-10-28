import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

llm = pd.read_csv("data/resultats_llm.csv")
bert = pd.read_csv("data/resultats_bert.csv")

# On ne garde que les avis communs aux deux (ceux utilisés par Mistral)
bert_subset = bert[bert["review"].isin(llm["review"])].dropna(subset=["pred"])

print(f"Avis Mistral : {len(llm)}")
print(f"Avis BERT (même sous-échantillon, hors neutres) : {len(bert_subset)}")

print("\n=== MISTRAL ===")
llm["pred"] = llm["pred_brut"].apply(lambda p: 1 if "positif" in p.lower() else 0)
print(f"Accuracy : {accuracy_score(llm['label'], llm['pred']):.3f}")
print(classification_report(llm["label"], llm["pred"], target_names=["négatif", "positif"]))

print("\n=== BERT (même sous-échantillon) ===")
print(f"Accuracy : {accuracy_score(bert_subset['label'], bert_subset['pred']):.3f}")
print(classification_report(bert_subset["label"], bert_subset["pred"], target_names=["négatif", "positif"]))