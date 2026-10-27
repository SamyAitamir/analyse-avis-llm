"""Point 2 et 3 (partie 120 avis) : isoler l'effet du prompt.

Sur les 120 avis d'origine (data/resultats_llm.csv, data/resultats_llm_stars.csv),
comparés à BERT (data/resultats_bert.csv) :
  (a) Mistral binaire restreint aux avis que Mistral 1-5 ne note pas 3 ;
  (b) Mistral binaire, Mistral 1-5 et BERT sur le sous-ensemble commun
      (avis non neutres pour Mistral 1-5 ET pour BERT) ;
  + IC 95 % de Wilson et tests de McNemar appariés.
Aucune inférence : lecture seule des CSV existants.
Sorties : resultats/13_accuracies_120.csv, resultats/13_tests_120.csv
"""
import os

import pandas as pd

from outils_eval import afficher_tableaux, normaliser_binaire, note_bert, parse_note, tableau_effet_prompt

os.makedirs("resultats", exist_ok=True)

ech = pd.read_csv("data/allocine_sample.csv")
id_par_avis = {r: i for i, r in enumerate(ech["review"])}
assert len(id_par_avis) == len(ech), "avis en double dans l'échantillon"

llm = pd.read_csv("data/resultats_llm.csv")
stars = pd.read_csv("data/resultats_llm_stars.csv")
bert = pd.read_csv("data/resultats_bert.csv")
assert (llm["review"].values == stars["review"].values).all(), "les 120 avis diffèrent entre 02 et 08"

df = pd.DataFrame({
    "id": llm["review"].map(id_par_avis),
    "label": llm["label"],
    "m_bin": llm["pred_brut"].apply(normaliser_binaire),
    "m_note": stars["note_brute"].apply(parse_note),
})
df["b_note"] = df["id"].map(dict(zip(range(len(bert)), bert["note_brute"].apply(note_bert))))
assert (bert["review"].values == ech["review"].values).all()
assert df["id"].notna().all() and df["b_note"].notna().all()

acc, tests, eff = tableau_effet_prompt(df)
print("=== 120 avis d'origine (passage unique des scripts 02 et 08) ===\n")
afficher_tableaux(acc, tests, eff)
acc.to_csv("resultats/13_accuracies_120.csv", index=False)
tests.to_csv("resultats/13_tests_120.csv", index=False)
