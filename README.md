# Analyse d'avis en français avec un LLM local

Ce projet compare, en exécution 100 % locale sur une GTX 1060 (6 Go de VRAM), un LLM généraliste (Mistral 7B quantifié, servi par Ollama) et un modèle spécialisé (BERT multilingue `nlptown`) pour analyser des avis en français. Il traite deux tâches : la classification de sentiment (positif / négatif) et l'extraction de thèmes, où KeyBERT est comparé à une catégorisation par LLM. Les critiques de films du corpus public Allociné servent de substitut à des verbatims clients.

Le projet est passé par plusieurs itérations. La première comparaison entre les deux modèles était biaisée : BERT pouvait écarter les avis qu'il notait 3 sur 5, alors que Mistral devait trancher sur tous. J'ai donc donné à Mistral la même échelle 1-5, puis comparé toutes les méthodes sur les mêmes avis, avec intervalles de confiance et tests de McNemar, d'abord sur 120 avis puis sur 500. Les thèmes ont été étudiés avec trois versions de KeyBERT, puis avec une catégorisation par Mistral croisée avec le sentiment. Une analyse manuelle des erreurs clôt l'ensemble.

**Pour une lecture rapide, ouvrir [`analyse_avis.ipynb`](analyse_avis.ipynb)** : démarche, code et résultats y sont réunis, avec leurs sorties enregistrées. Les données Allociné ne sont pas versionnées (`data/`).

## Fichiers

| Fichier | Rôle |
|---|---|
| `analyse_avis.ipynb` | Synthèse lisible : démarche, code, résultats et limites |
| `run_all.py` | Enchaîne les scripts dans l'ordre |
| `scripts/01_dataset.py` | Chargement d'Allociné, nettoyage, échantillon de 500 avis (250 négatifs, 250 positifs) |
| `scripts/02_sentiment_llm.py`, `scripts/03_evaluation_llm.py` | Mistral, prompt binaire, 120 avis, et son évaluation |
| `scripts/04_sentiment_bert.py`, `scripts/05_evaluation_bert.py` | BERT nlptown (notes 1-5), et son évaluation |
| `scripts/06_comparaison.py` | Première comparaison Mistral / BERT (celle qui était biaisée) |
| `scripts/08_sentiment_llm_stars.py`, `scripts/09_evaluation_llm_stars.py` | Mistral avec l'échelle 1-5, et son évaluation |
| `scripts/10_keywords.py`, `scripts/10_keywords_v2.py` | KeyBERT sur le texte concaténé (v1, v2) |
| `scripts/11_keywords_v3.py` | KeyBERT avis par avis (v3) |
| `scripts/12_themes_llm.py` | Catégorisation de thèmes par Mistral, 30 avis |
| `scripts/13_effet_prompt.py` | Isole l'effet du prompt à périmètre égal (IC, McNemar), 120 avis |
| `scripts/14_mistral_500.py` | Mistral binaire et 1-5 sur 500 avis (reprise possible) |
| `scripts/15_themes_llm_500.py` | Catégorisation de thèmes par Mistral, 500 avis |
| `scripts/16_evaluation_500.py` | Évaluation sur 500 avis : IC, McNemar, bootstrap, stabilité |
| `scripts/17_analyse_themes.py` | Normalisation des thèmes, croisement avec le sentiment, comparaison avec KeyBERT |
| `scripts/18_analyse_erreurs.py` | Analyse des erreurs des deux modèles |
| `scripts/19_figures.py` | Génération des graphiques |
| `scripts/outils_eval.py` | Fonctions statistiques communes (Wilson, McNemar, bootstrap) |
| `scripts/voir_erreur.py` | Inspection rapide des erreurs du prompt binaire |
| `resultats/` | Tableaux produits par les analyses, dont l'annotation manuelle des erreurs |
| `figures/` | Graphiques (PNG et PDF) |
| `mesures/` | Mesures annexes : mémoire GPU, temps d'inférence, troncature de KeyBERT, relance de Mistral |
| `requirements.txt` | Dépendances Python |

Le numéro 07 n'existe pas. `scripts/12_themes_llm.py` s'appelait `11_themes_llm.py` avant d'être renommé pour éviter un doublon.

## Reproduire

Prérequis : Python 3.10, [Ollama](https://ollama.com) avec le modèle `mistral`, une carte NVIDIA pour des temps raisonnables.

```bash
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python run_all.py --tout    # toute la chaîne, environ 40 minutes
```

Les scripts lisent et écrivent des chemins relatifs à la racine du projet : pour un script isolé, le lancer depuis la racine, par exemple `python scripts/16_evaluation_500.py`.
