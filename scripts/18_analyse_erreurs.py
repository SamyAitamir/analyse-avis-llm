"""Point 5 : analyse d'erreurs sur les 500 avis.

1. Extrait les avis mal classés (hors neutres) par Mistral 1-5 (14_mistral_500.py)
   et par BERT (04_sentiment_bert.py) -> resultats/18_erreurs.csv
2. Affiche les effectifs (faux positifs / faux négatifs, recouvrement entre modèles)
   et quelques exemples.
3. Si l'annotation manuelle resultats/18_typologie_erreurs.csv existe, affiche la
   répartition des types d'erreurs par modèle. Cette annotation est faite à la main,
   à la lecture des avis : c'est un jugement, une seule catégorie principale par erreur.
"""
import os
import textwrap

import pandas as pd

from outils_eval import binariser, note_bert, parse_note

os.makedirs("resultats", exist_ok=True)

ech = pd.read_csv("data/allocine_sample.csv")
ech["id"] = range(len(ech))
bert = pd.read_csv("data/resultats_bert.csv")
ms = pd.read_csv("data/resultats_llm_stars_500.csv").drop_duplicates("id").set_index("id")

df = ech.copy()
df["note_bert"] = bert["note_brute"].apply(note_bert).values
df["note_mistral"] = df["id"].map(ms["note_brute"].apply(parse_note))
df["bin_bert"] = df["note_bert"].apply(binariser)
df["bin_mistral"] = df["note_mistral"].apply(binariser)
df["n_mots"] = df["review"].str.split().str.len()

err_m = df["bin_mistral"].notna() & (df["bin_mistral"] != df["label"])
err_b = df["bin_bert"].notna() & (df["bin_bert"] != df["label"])
df["erreur_mistral"], df["erreur_bert"] = err_m, err_b

erreurs = df[err_m | err_b].copy()
erreurs["modele"] = erreurs.apply(lambda r: "les deux" if r.erreur_mistral and r.erreur_bert
                                  else ("Mistral 1-5" if r.erreur_mistral else "BERT"), axis=1)
erreurs[["id", "label", "note_mistral", "note_bert", "modele", "n_mots", "review"]].to_csv(
    "resultats/18_erreurs.csv", index=False)

print("=== Effectifs des erreurs (avis non neutres pour le modèle concerné) ===")
for nom, e, col in [("Mistral 1-5", err_m, "bin_mistral"), ("BERT", err_b, "bin_bert")]:
    tranches = df[col].notna().sum()
    fp = (e & (df["label"] == 0)).sum()
    fn = (e & (df["label"] == 1)).sum()
    print(f"{nom:12s}: {e.sum()} erreurs sur {tranches} avis tranchés "
          f"(négatifs classés positifs : {fp}, positifs classés négatifs : {fn})")
commun = df["bin_mistral"].notna() & df["bin_bert"].notna()
print(f"Sur les {commun.sum()} avis tranchés par les deux : erreurs Mistral seul "
      f"{(err_m & ~err_b & commun).sum()}, BERT seul {(err_b & ~err_m & commun).sum()}, "
      f"les deux {(err_m & err_b).sum()}")
print(f"Longueur médiane (mots) : erreurs {erreurs['n_mots'].median():.0f}, "
      f"tous les avis {df['n_mots'].median():.0f}")


def verdict(b, label):
    return "neutre (3)" if pd.isna(b) else ("juste" if b == label else "faux")


for nom, e, autre, col in [("Mistral 1-5", err_m, "BERT", "bin_bert"), ("BERT", err_b, "Mistral 1-5", "bin_mistral")]:
    v = df[e].apply(lambda r: verdict(r[col], r["label"]), axis=1).value_counts().to_dict()
    print(f"Sur les {e.sum()} erreurs de {nom}, verdict de {autre} : {v}")


def montrer(sous, titre, n=4):
    print(f"\n--- {titre} ---")
    for r in sous.head(n).itertuples():
        print(f"[id {r.id}] label={'positif' if r.label else 'négatif'}, "
              f"Mistral={r.note_mistral:.0f}, BERT={r.note_bert}")
        print(textwrap.fill(r.review[:420] + ("…" if len(r.review) > 420 else ""), 100,
                            initial_indent="   ", subsequent_indent="   "))


montrer(erreurs[erreurs["modele"] == "les deux"], "Erreurs communes aux deux modèles")
montrer(erreurs[erreurs["modele"] == "Mistral 1-5"], "Erreurs de Mistral 1-5 seul")
montrer(erreurs[erreurs["modele"] == "BERT"], "Erreurs de BERT seul")

# ---------------------------------------------------------------- typologie (annotation manuelle)
f = "resultats/18_typologie_erreurs.csv"
if os.path.exists(f):
    typo = pd.read_csv(f)
    manquants = set(erreurs["id"]) - set(typo["id"])
    if manquants:
        print(f"\nATTENTION : {len(manquants)} erreurs non annotées : {sorted(manquants)}")
    t = erreurs.merge(typo[["id", "type"]], on="id", how="left")
    lignes = []
    for nom, masque in [("Mistral 1-5", t["erreur_mistral"]), ("BERT", t["erreur_bert"])]:
        c = t[masque]["type"].value_counts()
        for typ, k in c.items():
            lignes.append({"modele": nom, "type": typ, "n": k})
    tab = pd.DataFrame(lignes).pivot_table(index="type", columns="modele", values="n",
                                           fill_value=0, aggfunc="sum")
    tab["total (avis distincts)"] = t.groupby("type")["id"].nunique()
    tab = tab.sort_values("total (avis distincts)", ascending=False)
    print("\n=== Typologie des erreurs (annotation manuelle, une catégorie par avis) ===")
    print(tab.to_string())
    tab.to_csv("resultats/18_typologie_comptes.csv")
