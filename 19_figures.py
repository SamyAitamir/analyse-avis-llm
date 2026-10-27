"""Point 6 : graphiques (PNG + PDF dans figures/).

  figures/notes_1_5_mistral_vs_bert   répartition des notes 1-5 (500 avis), par label réel
  figures/accuracy_ic95               accuracies et IC 95 % de Wilson, 120 avis vs 500 avis
  figures/themes_par_sentiment        thèmes attribués par Mistral, avis négatifs vs positifs

Pré-requis : 13_effet_prompt.py, 16_evaluation_500.py, 17_analyse_themes.py.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from outils_eval import note_bert, parse_note

os.makedirs("figures", exist_ok=True)

BLEU, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
ENCRE, ENCRE2, MUET, GRILLE, AXE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXE, "axes.linewidth": 0.8,
    "axes.labelcolor": ENCRE2, "xtick.color": MUET, "ytick.color": MUET,
    "xtick.labelcolor": ENCRE2, "ytick.labelcolor": ENCRE2,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def pct(x, d=1):
    return f"{100 * x:.{d}f}".replace(".", ",") + " %"


def sauver(fig, nom):
    fig.savefig(f"figures/{nom}.png", dpi=200)
    fig.savefig(f"figures/{nom}.pdf")
    plt.close(fig)
    print("figures/" + nom + ".png")


# ---------------------------------------------------------------- 1. notes 1-5
ech = pd.read_csv("data/allocine_sample.csv")
bert = pd.read_csv("data/resultats_bert.csv")
ms = pd.read_csv("data/resultats_llm_stars_500.csv").drop_duplicates("id").set_index("id").sort_index()
notes = pd.DataFrame({"label": ech["label"],
                      "bert": bert["note_brute"].apply(note_bert),
                      "mistral": ms["note_brute"].apply(parse_note).reindex(range(len(ech))).values})

fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
w = 0.38
x = np.arange(1, 6)
for ax, lab, titre in [(axes[0], 0, "Avis réellement négatifs"), (axes[1], 1, "Avis réellement positifs")]:
    sub = notes[notes["label"] == lab]
    for dx, col, nom, couleur in [(-w / 2 - 0.01, "bert", "BERT nlptown", BLEU),
                                  (w / 2 + 0.01, "mistral", "Mistral 7B (prompt 1-5)", AQUA)]:
        c = sub[col].value_counts().reindex(range(1, 6), fill_value=0)
        ax.bar(x + dx, c.values, w, color=couleur, label=nom)
        for xi, v in zip(x + dx, c.values):
            ax.text(xi, v + 1.5, str(v), ha="center", va="bottom", fontsize=7, color=ENCRE2)
    ax.axvspan(2.5, 3.5, color="#f0efec", zorder=0)
    ax.set_xticks(x)
    ax.set_xlabel("Note prédite (3 = exclue de l'évaluation)")
    ax.set_title(f"{titre} (n = {len(sub)})", fontsize=9, color=ENCRE, loc="left")
    ax.yaxis.grid(True, color=GRILLE, lw=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("Nombre d'avis")
axes[1].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
sauver(fig, "notes_1_5_mistral_vs_bert")

# ---------------------------------------------------------------- 2. accuracies + IC
a120 = pd.read_csv("resultats/13_accuracies_120.csv")
a500 = pd.read_csv("resultats/16_accuracies_500.csv")
etiquettes = [f"{p}\n{m}" for p, m in zip(a120["perimetre"], a120["methode"])]
etiquettes = [e.replace("Tous les avis exploitables", "Tous les avis")
               .replace("Avis non neutres pour", "Non neutres pour") for e in etiquettes]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
k = len(a120)
for dy, tab, couleur, nom in [(0.17, a120, ORANGE, "120 avis (premier sous-échantillon)"),
                              (-0.17, a500, BLEU, "500 avis")]:
    for i, r in enumerate(tab.itertuples()):
        y = k - 1 - i + dy
        ax.plot([r.ic95_bas, r.ic95_haut], [y, y], color=couleur, lw=2, solid_capstyle="round")
        ax.plot(r.accuracy, y, "o", ms=6.5, color=couleur, mec="#fcfcfb", mew=1.5, zorder=3,
                label=nom if i == 0 else None)
        ax.text(1.003, y, f"{pct(r.accuracy)} ({r.corrects}/{r.n})", va="center", fontsize=6.8,
                color=ENCRE2, transform=ax.get_yaxis_transform())
for i in range(1, k):
    ax.axhline(k - 1 - i + 0.5, color=GRILLE, lw=0.6)
ax.set_yticks(range(k))
ax.set_yticklabels(etiquettes[::-1], fontsize=7.2)
ax.set_xlim(0.80, 1.0)
ax.xaxis.set_major_locator(mticker.MultipleLocator(0.05))
ax.set_ylim(-0.6, k - 0.4)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: pct(v, 0)))
ax.set_xlabel("Accuracy sur le périmètre indiqué, avec IC 95 % de Wilson")
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
fig.subplots_adjust(left=0.30, right=0.84, top=0.93, bottom=0.10)
sauver(fig, "accuracy_ic95")

# ---------------------------------------------------------------- 3. thèmes x sentiment
t = pd.read_csv("resultats/17_themes_x_sentiment.csv")
t = t.iloc[::-1]
fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(t) + 1.0))
y = np.arange(len(t))
h = 0.38
for dy, col, couleur, nom in [(h / 2 + 0.01, "pct_neg", ORANGE, "avis négatifs"),
                              (-h / 2 - 0.01, "pct_pos", BLEU, "avis positifs")]:
    ax.barh(y + dy, t[col], height=h, color=couleur, label=nom)
    for yi, v, n in zip(y + dy, t[col], t[col.replace("pct", "n")]):
        ax.text(v + 0.004, yi, f"{pct(v, 0)} ({n})", va="center", fontsize=6.8, color=ENCRE2)
ax.set_yticks(y)
ax.set_yticklabels(t["theme"], fontsize=8)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: pct(v, 0)))
ax.set_xlabel("Part des avis de chaque polarité (n = 250 + 250)")
ax.xaxis.grid(True, color=GRILLE, lw=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(loc="lower right", fontsize=7.5)
fig.tight_layout()
sauver(fig, "themes_par_sentiment")
