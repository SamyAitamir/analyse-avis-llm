"""Point 4 : analyse des thèmes attribués par Mistral (15_themes_llm_500.py, 500 avis).

1. Normalisation des réponses, en trois niveaux, et taux de réponses hors liste :
   - conforme brut : la réponse est exactement une des 7 catégories ;
   - normalisation de surface : casse, accents, articles, guillemets, ponctuation,
     préfixe « catégorie : », explication entre parenthèses ;
   - rattachement par synonyme : libellé inventé mais sans ambiguïté
     (ex. « histoire » -> scénario, « jeu des acteurs » -> jeu d'acteur) ;
   le reste est classé « plusieurs catégories » ou « hors liste ».
2. Croisement thème x sentiment réel (label) + test du khi-deux.
3. Éléments de comparaison avec KeyBERT v2/v3.

Sorties : resultats/17_themes_normalises.csv, resultats/17_themes_x_sentiment.csv,
          resultats/17_hors_liste.csv
"""
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

os.makedirs("resultats", exist_ok=True)

CATS = ["scénario", "jeu d'acteur", "réalisation/mise en scène", "rythme", "humour", "émotion", "autre"]


def sans_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


CANON = {sans_accents(c): c for c in CATS}
# Synonymes sans ambiguïté (forme sans accents -> catégorie). Volontairement courte.
SYNONYMES = {
    "scenario": "scénario", "histoire": "scénario", "intrigue": "scénario", "scenario/histoire": "scénario",
    "jeu d'acteur": "jeu d'acteur", "jeu d'acteurs": "jeu d'acteur", "jeu des acteurs": "jeu d'acteur",
    "acteur": "jeu d'acteur", "acteurs": "jeu d'acteur", "acteur/jeu d'acteur": "jeu d'acteur",
    "acteur (jeu d'acteur)": "jeu d'acteur", "interpretation": "jeu d'acteur", "casting": "jeu d'acteur",
    "realisation": "réalisation/mise en scène", "mise en scene": "réalisation/mise en scène",
    "realisation/mise en scene": "réalisation/mise en scène",
    "rythme": "rythme", "humour": "humour", "emotion": "émotion", "emotions": "émotion", "autre": "autre",
}
# Mots-clés servant à repérer les catégories citées dans une réponse longue ou multiple
MOTS_CLES = {
    "scénario": ["scenario", "histoire", "intrigue"],
    "jeu d'acteur": ["jeu d'acteur", "jeu des acteurs", "acteur"],
    "réalisation/mise en scène": ["realisation", "mise en scene"],
    "rythme": ["rythme"],
    "humour": ["humour"],
    "émotion": ["emotion"],
    "autre": ["autre"],
}


def surface(rep):
    s = str(rep).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(la |le )?(categorie|catégorie)( principale)?( est)?\s*:?\s*", "", s)
    s = re.sub(r"\(.*?\)", "", s) if not s.startswith("acteur (") else s
    s = s.strip(" .:;!\"'«»*`")
    s = re.sub(r"^(le |la |les |l'|l’|un |une |du |des |de la )", "", s)
    return sans_accents(s.strip(" .:;!\"'«»*`"))


def categories_citees(s):
    return [c for c, mots in MOTS_CLES.items() if any(m in s for m in mots)]


def normaliser(rep):
    """Retourne (catégorie finale, statut)."""
    rep = str(rep)
    if rep in CATS:
        return rep, "conforme brut"
    s = surface(rep)
    if s in CANON:
        return CANON[s], "conforme après normalisation de surface"
    if s in SYNONYMES:
        return SYNONYMES[s], "rattaché par synonyme"
    citees = categories_citees(s)
    if len(citees) > 1:
        return "plusieurs catégories", "plusieurs catégories"
    if len(citees) == 1:
        return citees[0], "rattaché par synonyme"
    return "hors liste", "hors liste"


df = pd.read_csv("data/resultats_themes_llm_500.csv").drop_duplicates("id").sort_values("id")
print(f"{len(df)} avis catégorisés (labels : {df['label'].value_counts().sort_index().to_dict()})")
if len(df) < 500:
    print("ATTENTION : traitement incomplet, relancer scripts/15_themes_llm_500.py")
res = df["theme"].apply(normaliser)
df["categorie"] = [r[0] for r in res]
df["statut"] = [r[1] for r in res]
df[["id", "label", "theme", "categorie", "statut"]].to_csv("resultats/17_themes_normalises.csv", index=False)

# ---------------------------------------------------------------- 1. conformité
print("\n=== Conformité des réponses ===")
statuts = df["statut"].value_counts()
for s in ["conforme brut", "conforme après normalisation de surface", "rattaché par synonyme",
          "plusieurs catégories", "hors liste"]:
    k = statuts.get(s, 0)
    print(f"  {s:42s} {k:4d}  ({100 * k / len(df):5.1f} %)")
non_brut = (df["statut"] != "conforme brut").mean()
print(f"Taux de réponses hors liste au sens strict (pas exactement une catégorie) : {100 * non_brut:.1f} %")
print(f"Taux après normalisation de surface : "
      f"{100 * df['statut'].isin(['rattaché par synonyme', 'plusieurs catégories', 'hors liste']).mean():.1f} %")
print(f"Taux après rattachement des synonymes (hors liste + plusieurs catégories) : "
      f"{100 * df['statut'].isin(['plusieurs catégories', 'hors liste']).mean():.1f} %")
hl = df[df["statut"] != "conforme brut"].groupby(["theme", "categorie", "statut"]).size() \
    .reset_index(name="n").sort_values("n", ascending=False)
hl.to_csv("resultats/17_hors_liste.csv", index=False)
print("\nRéponses non conformes (brut -> catégorie retenue), les plus fréquentes :")
for r in hl.head(25).itertuples():
    print(f"  {r.n:3d} x {r.theme[:70]!r:74s} -> {r.categorie}")

# ---------------------------------------------------------------- 2. thème x sentiment
print("\n=== Thème x sentiment réel ===")
ordre = CATS + ["plusieurs catégories", "hors liste"]
ct = pd.crosstab(df["categorie"], df["label"]).reindex(ordre, fill_value=0)
ct.columns = ["n_neg", "n_pos"]
tot = df["label"].value_counts().sort_index().values
ct["pct_neg"] = ct["n_neg"] / tot[0]
ct["pct_pos"] = ct["n_pos"] / tot[1]
ct["part_neg_dans_theme"] = ct["n_neg"] / (ct["n_neg"] + ct["n_pos"]).replace(0, np.nan)
ct = ct[(ct["n_neg"] + ct["n_pos"]) > 0]
print(ct.assign(pct_neg=(100 * ct["pct_neg"]).round(1), pct_pos=(100 * ct["pct_pos"]).round(1),
                part_neg_dans_theme=(100 * ct["part_neg_dans_theme"]).round(0)).to_string())
ct.reset_index().rename(columns={"categorie": "theme"}).to_csv("resultats/17_themes_x_sentiment.csv", index=False)

testables = ct[(ct["n_neg"] + ct["n_pos"]) >= 10][["n_neg", "n_pos"]]
chi2, p, ddl, attendus = chi2_contingency(testables.values)
v = np.sqrt(chi2 / (testables.values.sum() * (min(testables.shape) - 1)))
print(f"\nKhi-deux d'indépendance thème x sentiment (thèmes avec >= 10 avis : {list(testables.index)}) : "
      f"chi2 = {chi2:.1f}, ddl = {ddl}, p = {p:.2g}, V de Cramér = {v:.2f}")
if (attendus < 5).any():
    print("  (certains effectifs attendus < 5 : test indicatif)")

# Stabilité sur les 30 avis du test d'origine (12_themes_llm.py, sans seed)
orig = pd.read_csv("data/resultats_themes_llm.csv")
ech = pd.read_csv("data/allocine_sample.csv")
id_par_avis = {r: i for i, r in enumerate(ech["review"])}
orig["id"] = orig["review"].map(id_par_avis)
orig["categorie"] = orig["theme"].apply(lambda r: normaliser(r)[0])
m = orig.merge(df[["id", "categorie"]], on="id", suffixes=("_orig", "_500"))
print(f"\nAccord avec le passage d'origine sur les {len(m)} avis communs (après normalisation) : "
      f"{(m['categorie_orig'] == m['categorie_500']).sum()}/{len(m)}")

# ---------------------------------------------------------------- 3. comparaison KeyBERT
print("\n=== Éléments de comparaison avec KeyBERT ===")
f_neg = "mesures/resultats/keybert_v3_distribution_negatifs.csv"
f_pos = "mesures/resultats/keybert_v3_distribution_positifs.csv"
if os.path.exists(f_neg) and os.path.exists(f_pos):
    # Distribution complète de KeyBERT v3 (reproduit 11_keywords_v3.py, cf. mesures/keybert_distribution.py)
    lex = {
        "scénario": r"\bscenar|\bhistoire|\bintrigue|\bscript",
        "jeu d'acteur": r"\bacteur|\bactrice|\binterpret|\bcasting|\bjeu\b",
        "réalisation/mise en scène": r"\brealisat|mise en scene|\bcamera|\bphotograph|\bimages?\b|\bplans?\b",
        "rythme": r"\brythme|\blent\b|\blenteur|\blongueurs?\b|\bennui|\bennuy",
        "humour": r"\bhumour|\bdrole|\brires?\b|\bgags?\b|\bcomedie",
        "émotion": r"\bemotion|\bemouv|\btouchant|\blarmes?\b",
    }
    for pol, f in [("négatifs", f_neg), ("positifs", f_pos)]:
        kb = pd.read_csv(f)
        kb["sa"] = kb["terme"].astype(str).apply(sans_accents)
        kb["cat"] = kb["sa"].apply(lambda t: next((c for c, p in lex.items() if re.search(p, t)), None))
        total = kb["occurrences"].sum()
        couvert = kb.loc[kb["cat"].notna(), "occurrences"].sum()
        print(f"KeyBERT v3, avis {pol} : {total} expressions extraites, {len(kb)} distinctes ; "
              f"{100 * couvert / total:.1f} % contiennent un mot rattachable à une catégorie "
              f"(lexique indicatif) ; top 10 : {', '.join(kb['terme'].head(10))}")
else:
    print("Distribution KeyBERT v3 absente (lancer mesures/keybert_distribution.py).")
for pol, lab in [("négatifs", 0), ("positifs", 1)]:
    top = df[df["label"] == lab]["categorie"].value_counts().head(4)
    print(f"Mistral, avis {pol} : " + ", ".join(f"{c} {100 * n / tot[lab]:.0f} %" for c, n in top.items()))
