"""Fonctions communes aux scripts d'évaluation 13_ à 19_ :
parsing des sorties, binarisation, intervalles de confiance, test de McNemar."""
import re
from math import comb, sqrt

import numpy as np


# ------------------------------------------------------------ parsing des sorties
def normaliser_binaire(pred):
    """Même règle que 03_evaluation_llm.py : 1 = positif, 0 = négatif, NaN sinon."""
    p = str(pred).lower()
    if "positif" in p:
        return 1
    if "négatif" in p or "negatif" in p:
        return 0
    return np.nan


def parse_note(s):
    """Même règle que 09_evaluation_llm_stars.py : premier chiffre entre 1 et 5."""
    m = re.search(r"[1-5]", str(s))
    return int(m.group()) if m else np.nan


def note_bert(label):
    """'4 stars' -> 4 (sortie du pipeline nlptown)."""
    return int(str(label)[0])


def binariser(note):
    """1-2 -> 0, 4-5 -> 1, 3 ou non parsable -> NaN (exclu)."""
    if note in (1, 2):
        return 0
    if note in (4, 5):
        return 1
    return np.nan


# ------------------------------------------------------------ statistiques
def wilson(k, n, z=1.959964):
    """Intervalle de confiance de Wilson à 95 % pour une proportion k/n."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """Test de McNemar exact (binomial bilatéral) sur b et c, les paires discordantes."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def accuracy(y, pred):
    """Retourne (corrects, n, accuracy, (borne basse, borne haute))."""
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    k, n = int((y == pred).sum()), len(y)
    return k, n, (k / n if n else np.nan), wilson(k, n)


def comparer_apparie(y, pred_a, pred_b):
    """McNemar entre deux prédicteurs sur les mêmes exemples.
    Retourne (a juste & b faux, a faux & b juste, p-valeur)."""
    y, a, b = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    ok_a, ok_b = a == y, b == y
    n10, n01 = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    return n10, n01, mcnemar_exact(n10, n01)


def fmt_acc(k, n, acc, ic):
    return f"{100 * acc:5.1f} % ({k}/{n})  IC95 [{100 * ic[0]:.1f} ; {100 * ic[1]:.1f}]"


# ------------------------------------------------------------ analyse commune 120 / 500
def tableau_effet_prompt(df):
    """df : colonnes label, m_bin (0/1/NaN), m_note (1-5/NaN), b_note (1-5).
    Retourne (tableau des accuracies, tableau des tests appariés, effectifs)."""
    import pandas as pd

    d = df.copy()
    d["m_note_bin"] = d["m_note"].apply(binariser)
    d["b_note_bin"] = d["b_note"].apply(binariser)
    tranche_m = d["m_note_bin"].notna()
    tranche_b = d["b_note_bin"].notna()
    commun = tranche_m & tranche_b & d["m_bin"].notna()
    n = len(d)

    lignes = []

    def ajoute(perimetre, methode, masque, col):
        k, nn, acc, ic = accuracy(d.loc[masque, "label"], d.loc[masque, col])
        lignes.append({"perimetre": perimetre, "methode": methode, "n": nn, "corrects": k,
                       "accuracy": acc, "ic95_bas": ic[0], "ic95_haut": ic[1],
                       "couverture": nn / n})

    tous_m = d["m_bin"].notna()
    ajoute(f"Tous les avis exploitables", "Mistral binaire", tous_m, "m_bin")
    ajoute(f"Avis non neutres pour Mistral 1-5", "Mistral 1-5", tranche_m, "m_note_bin")
    ajoute(f"Avis non neutres pour BERT", "BERT 1-5", tranche_b, "b_note_bin")
    ajoute(f"(a) Avis non neutres pour Mistral 1-5", "Mistral binaire", tranche_m & tous_m, "m_bin")
    ajoute(f"(b) Sous-ensemble commun", "Mistral binaire", commun, "m_bin")
    ajoute(f"(b) Sous-ensemble commun", "Mistral 1-5", commun, "m_note_bin")
    ajoute(f"(b) Sous-ensemble commun", "BERT 1-5", commun, "b_note_bin")
    acc = pd.DataFrame(lignes)

    tests = []
    for perimetre, masque, a, b, na, nb in [
        ("(a) Avis non neutres pour Mistral 1-5", tranche_m & tous_m, "m_note_bin", "m_bin", "Mistral 1-5", "Mistral binaire"),
        ("(b) Sous-ensemble commun", commun, "m_note_bin", "m_bin", "Mistral 1-5", "Mistral binaire"),
        ("(b) Sous-ensemble commun", commun, "m_note_bin", "b_note_bin", "Mistral 1-5", "BERT 1-5"),
        ("(b) Sous-ensemble commun", commun, "m_bin", "b_note_bin", "Mistral binaire", "BERT 1-5"),
    ]:
        sub = d[masque]
        n10, n01, p = comparer_apparie(sub["label"], sub[a], sub[b])
        tests.append({"perimetre": perimetre, "A": na, "B": nb, "n": len(sub),
                      "A_juste_B_faux": n10, "A_faux_B_juste": n01, "p_mcnemar": p})
    tests = pd.DataFrame(tests)

    eff = {"n_total": n, "neutres_mistral_1_5": int((d["m_note"] == 3).sum()),
           "non_parsables_mistral_1_5": int(d["m_note"].isna().sum()),
           "non_exploitables_mistral_binaire": int(d["m_bin"].isna().sum()),
           "neutres_bert": int((d["b_note"] == 3).sum()),
           "neutres_communs": int(((d["m_note"] == 3) & (d["b_note"] == 3)).sum()),
           "sous_ensemble_commun": int(commun.sum())}
    return acc, tests, eff


def afficher_tableaux(acc, tests, eff):
    print("Effectifs :", eff)
    print()
    print(f"{'Périmètre':42s} {'Méthode':16s} {'Accuracy (corrects/n)  IC 95 %':>42s}  Couv.")
    for r in acc.itertuples():
        print(f"{r.perimetre:42s} {r.methode:16s} "
              f"{fmt_acc(r.corrects, r.n, r.accuracy, (r.ic95_bas, r.ic95_haut)):>42s}  {100 * r.couverture:4.0f} %")
    print()
    print("Tests de McNemar exacts (prédictions appariées) :")
    for r in tests.itertuples():
        print(f"  {r.perimetre:40s} {r.A} vs {r.B} (n={r.n}) : "
              f"A seul juste {r.A_juste_B_faux}, B seul juste {r.A_faux_B_juste}, p = {r.p_mcnemar:.3f}")


def bootstrap_diff(y, pred_a, pred_b, n_boot=10000, graine=0):
    """IC 95 % bootstrap (percentiles, rééchantillonnage apparié des avis)
    de la différence d'accuracy A - B."""
    rng = np.random.default_rng(graine)
    y, a, b = (np.asarray(v, dtype=float) for v in (y, pred_a, pred_b))
    ok = (a == y).astype(float) - (b == y).astype(float)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    diffs = ok[idx].mean(axis=1)
    return ok.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
