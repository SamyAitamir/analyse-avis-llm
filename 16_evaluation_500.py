"""Points 2 et 3 sur les 500 avis : refait l'analyse de 13_effet_prompt.py sur les
sorties de 14_mistral_500.py, et la compare aux résultats d'origine sur 120 avis
(conservés tels quels).

Sorties : resultats/16_accuracies_500.csv, resultats/16_tests_500.csv,
          resultats/16_accuracies_120_relance.csv, resultats/16_stabilite_120.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from outils_eval import (afficher_tableaux, binariser, bootstrap_diff, normaliser_binaire,
                         note_bert, parse_note, tableau_effet_prompt)

os.makedirs("resultats", exist_ok=True)

ech = pd.read_csv("data/allocine_sample.csv")
bert = pd.read_csv("data/resultats_bert.csv")
assert (bert["review"].values == ech["review"].values).all()

mb = pd.read_csv("data/resultats_llm_500.csv").drop_duplicates("id").set_index("id").sort_index()
ms = pd.read_csv("data/resultats_llm_stars_500.csv").drop_duplicates("id").set_index("id").sort_index()
if len(mb) < len(ech) or len(ms) < len(ech):
    print(f"ATTENTION : traitement incomplet (binaire {len(mb)}/500, 1-5 {len(ms)}/500). "
          "Relancer 14_mistral_500.py pour terminer.")
    if "--partiel" not in sys.argv:
        sys.exit(1)
ids = mb.index.intersection(ms.index)

df = pd.DataFrame({
    "id": ids,
    "label": ech.loc[ids, "label"].values,
    "m_bin": mb.loc[ids, "pred_brut"].apply(normaliser_binaire).values,
    "m_note": ms.loc[ids, "note_brute"].apply(parse_note).values,
    "b_note": bert.loc[ids, "note_brute"].apply(note_bert).values,
})
assert (mb.loc[ids, "label"].values == df["label"].values).all()

# ---------------------------------------------------------------- 500 avis
print(f"=== {len(df)} avis (14_mistral_500.py, seed 42) ===\n")
print("Réponses brutes Mistral 1-5 :", ms["note_brute"].astype(str).value_counts().to_dict())
non_conf = mb["pred_brut"][mb["pred_brut"].astype(str).str.split().str.len() > 1]
print(f"Réponses binaires de plus d'un mot : {len(non_conf)} (toutes interprétables : "
      f"{non_conf.apply(normaliser_binaire).notna().all()})")
deux = mb["pred_brut"].astype(str)
deux = deux[deux.str.contains("positif") & deux.str.contains("négatif|negatif")]
print(f"Réponses binaires citant à la fois 'positif' et 'négatif' : {len(deux)} "
      "(la règle de 03_evaluation_llm.py les classe 'positif')")
for i, rep in deux.items():
    print(f"   id {i}, label {ech.loc[i, 'label']} : {rep[:90]!r}")
print("Temps moyen par avis : binaire %.2f s, 1-5 %.2f s\n" % (mb["duree_s"].mean(), ms["duree_s"].mean()))
acc, tests, eff = tableau_effet_prompt(df)
afficher_tableaux(acc, tests, eff)
acc.to_csv("resultats/16_accuracies_500.csv", index=False)
tests.to_csv("resultats/16_tests_500.csv", index=False)

# IC bootstrap apparié de la différence sur le sous-ensemble commun
d = df.assign(m_note_bin=df["m_note"].apply(binariser), b_note_bin=df["b_note"].apply(binariser))
commun = d.dropna(subset=["m_note_bin", "b_note_bin", "m_bin"])
for a, b, na, nb in [("m_note_bin", "b_note_bin", "Mistral 1-5", "BERT"),
                     ("m_bin", "b_note_bin", "Mistral binaire", "BERT"),
                     ("m_note_bin", "m_bin", "Mistral 1-5", "Mistral binaire")]:
    diff, lo, hi = bootstrap_diff(commun["label"], commun[a], commun[b])
    print(f"Différence {na} - {nb} (sous-ensemble commun, n={len(commun)}) : "
          f"{100 * diff:+.1f} points, IC95 bootstrap [{100 * lo:+.1f} ; {100 * hi:+.1f}]")

# Accuracy stricte (avis exclus comptés comme erreurs) sur les 500
print("\nAccuracy stricte (neutres / non parsables = erreurs) :")
for nom, col in [("Mistral binaire", d["m_bin"]), ("Mistral 1-5", d["m_note_bin"]), ("BERT 1-5", d["b_note_bin"])]:
    print(f"  {nom:16s} {100 * (col == d['label']).mean():.1f} %")

# ---------------------------------------------------------------- comparaison avec les 120 d'origine
print("\n=== Les 120 avis d'origine, relancés dans 14_mistral_500.py ===\n")
id_par_avis = {r: i for i, r in enumerate(ech["review"])}
llm = pd.read_csv("data/resultats_llm.csv")
stars = pd.read_csv("data/resultats_llm_stars.csv")
ids120 = llm["review"].map(id_par_avis)
sous = df.set_index("id").loc[ids120].reset_index()
acc120, tests120, eff120 = tableau_effet_prompt(sous)
afficher_tableaux(acc120, tests120, eff120)
acc120.to_csv("resultats/16_accuracies_120_relance.csv", index=False)

orig_bin = llm["pred_brut"].apply(normaliser_binaire).values
orig_note = stars["note_brute"].apply(parse_note).values
stab = pd.DataFrame([{
    "accord_binaire": np.mean(orig_bin == sous["m_bin"].values),
    "accord_note_exacte": np.mean(orig_note == sous["m_note"].values),
    "accord_note_binarisee": np.mean(pd.Series(orig_note).apply(binariser).fillna(-1).values
                                     == sous["m_note"].apply(binariser).fillna(-1).values),
}])
print("\nAccord entre le passage d'origine (scripts 02/08) et la relance sur les 120 avis :")
print(stab.round(3).to_string(index=False))
stab.to_csv("resultats/16_stabilite_120.csv", index=False)
