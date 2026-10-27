"""Diagnostic de la troncature silencieuse (KeyBERT et BERT nlptown).

Mesure, sans relancer d'inférence, combien de tokens voient réellement
les modèles d'embeddings / de classification utilisés dans le projet.
"""
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from keybert import KeyBERT

df = pd.read_csv("data/allocine_sample.csv")
neg = df[df["label"] == 0]["review"].tolist()
pos = df[df["label"] == 1]["review"].tolist()

st = SentenceTransformer("all-MiniLM-L6-v2")  # modèle par défaut de KeyBERT()
assert KeyBERT().model.embedding_model.__class__.__name__ == "SentenceTransformer"
tok = st.tokenizer
max_len = st.max_seq_length

def n_tokens(t):
    return len(tok(t, add_special_tokens=True)["input_ids"])

out = {"keybert_modele_defaut": "all-MiniLM-L6-v2", "max_seq_length": max_len}
for nom, textes in [("negatifs", neg), ("positifs", pos)]:
    concat = " ".join(textes)
    nt = n_tokens(concat)
    # nombre d'avis entièrement contenus dans la fenêtre de 256 tokens
    cumul, n_avis_vus = 0, 0
    for t in textes:
        cumul += n_tokens(t) - 2
        if cumul <= max_len - 2:
            n_avis_vus += 1
        else:
            break
    out[nom] = {"n_caracteres": len(concat), "n_tokens": nt,
                "part_vue_pct": round(100 * max_len / nt, 2),
                "avis_entierement_vus": n_avis_vus}

lens = [n_tokens(t) for t in df["review"]]
out["avis_individuels_sup_256_tokens"] = int(sum(l > max_len for l in lens))
out["tokens_par_avis_mediane"] = float(pd.Series(lens).median())
out["tokens_par_avis_max"] = int(max(lens))

# BERT nlptown : limite 512 tokens
tok_bert = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
lens_b = [len(tok_bert(t)["input_ids"]) for t in df["review"]]
out["nlptown_avis_sup_512_tokens"] = int(sum(l > 512 for l in lens_b))
out["nlptown_tokens_max"] = int(max(lens_b))


with open("mesures/resultats/diagnostic_troncature.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
