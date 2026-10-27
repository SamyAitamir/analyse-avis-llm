"""Exécute les scripts du projet dans l'ordre.

Usage :
  python run_all.py              # analyses seulement : recalcule tableaux et figures
                                 # à partir des CSV existants (rapide, sans inférence)
  python run_all.py --tout       # toute la chaîne, inférences comprises (~40 min sur GTX 1060)
  python run_all.py --etapes 13 16 19   # étapes choisies (numéros ci-dessous)

Attention : avec --tout, les étapes 02, 04, 08 et 12 réécrivent leurs CSV dans data/
(les sorties de Mistral ne sont pas déterministes dans ces scripts d'origine : les
chiffres changeront légèrement). Les étapes 14 et 15 sont reprenables : si leurs CSV
sont complets, elles ne refont rien.

Une seule famille de modèles en VRAM à la fois : avant chaque étape qui charge un
modèle PyTorch (BERT, KeyBERT), Mistral est déchargé d'Ollama.
"""
import os
import subprocess
import sys
import time

# (numéro, script, catégorie) — catégories : donnees, mistral, torch, analyse
ETAPES = [
    ("01", "01_dataset.py", "donnees"),            # échantillon 250/250 -> data/allocine_sample.csv
    ("02", "02_sentiment_llm.py", "mistral"),       # Mistral binaire, 120 avis
    ("03", "03_evaluation_llm.py", "analyse"),
    ("04", "04_sentiment_bert.py", "torch"),        # BERT nlptown, 500 avis
    ("05", "05_evaluation_bert.py", "analyse"),
    ("06", "06_comparaison.py", "analyse"),         # comparaison de l'itération 1 (biaisée)
    ("08", "08_sentiment_llm_stars.py", "mistral"), # Mistral 1-5, mêmes 120 avis
    ("09", "09_evaluation_llm_stars.py", "analyse"),
    ("10", "10_keywords.py", "torch"),              # KeyBERT sur texte concaténé (v1)
    ("10v2", "10_keywords_v2.py", "torch"),         # + mots vides (v2)
    ("11", "11_keywords_v3.py", "torch"),           # KeyBERT par avis (v3)
    ("12", "12_themes_llm.py", "mistral"),          # thèmes par LLM, 30 avis
    ("13", "13_effet_prompt.py", "analyse"),        # isoler l'effet du prompt, 120 avis
    ("14", "14_mistral_500.py", "mistral"),         # Mistral binaire + 1-5, 500 avis (reprenable)
    ("15", "15_themes_llm_500.py", "mistral"),      # thèmes par LLM, 500 avis (reprenable)
    ("16", "16_evaluation_500.py", "analyse"),
    ("17", "17_analyse_themes.py", "analyse"),
    ("18", "18_analyse_erreurs.py", "analyse"),
    ("19", "19_figures.py", "analyse"),
    ("nb", "analyse_avis.ipynb", "analyse"),        # réexécute le notebook et enregistre ses sorties
]


def ollama_actif():
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def decharger_mistral():
    subprocess.run(["ollama", "stop", "mistral"], capture_output=True)


def main():
    args = sys.argv[1:]
    if "--etapes" in args:
        choix = set(args[args.index("--etapes") + 1:])
        etapes = [e for e in ETAPES if e[0] in choix]
    elif "--tout" in args:
        etapes = ETAPES
    else:
        etapes = [e for e in ETAPES if e[2] == "analyse"]

    if not args or "--etapes" in args:
        if not os.path.exists("data/resultats_llm_500.csv"):
            sys.exit("data/ est absent (non versionné). Lancer d'abord : python run_all.py --tout")

    if any(c == "mistral" for _, _, c in etapes) and not ollama_actif():
        sys.exit("Ollama ne répond pas : démarrer le serveur Ollama (et `ollama pull mistral`).")

    for num, script, cat in etapes:
        if cat == "torch":
            decharger_mistral()
        print(f"\n{'=' * 70}\n[{num}] {script}\n{'=' * 70}", flush=True)
        t0 = time.time()
        if script.endswith(".ipynb"):
            commande = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", script]
        else:
            commande = [sys.executable, script]
        r = subprocess.run(commande)
        if r.returncode != 0:
            sys.exit(f"Échec de {script} (code {r.returncode}).")
        print(f"[{num}] terminé en {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
