"""Mesure de l'occupation mémoire GPU des modèles utilisés (section « Choix des modèles »)."""
import json, subprocess
import ollama, pandas as pd, torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer

def vram_mio():
    return int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"]).decode().split()[0])

out = {"vram_totale_mio": int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total",
                                                      "--format=csv,noheader,nounits"]).decode().split()[0])}
out["vram_repos_mio"] = vram_mio()
ollama.chat(model="mistral", messages=[{"role": "user", "content": "Réponds OK"}])
out["vram_mistral_charge_mio"] = vram_mio()
ps = ollama.ps()
m = ps["models"][0]
out["ollama_ps"] = {"taille_octets": m["size"], "taille_vram_octets": m["size_vram"],
                    "contexte": m.get("context_length")}
subprocess.run(["ollama", "stop", "mistral"], check=True)

clf = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=0)
out["bert_parametres"] = sum(p.numel() for p in clf.model.parameters())
out["bert_dtype"] = str(next(clf.model.parameters()).dtype)
torch.cuda.reset_peak_memory_stats()
clf(pd.read_csv("data/allocine_sample.csv")["review"].tolist(), truncation=True, batch_size=16)
out["bert_poids_mio"] = round(torch.cuda.memory_allocated() / 2**20)
out["bert_pic_inference_mio"] = round(torch.cuda.max_memory_allocated() / 2**20)
out["minilm_parametres"] = sum(p.numel() for p in SentenceTransformer("all-MiniLM-L6-v2").parameters())
json.dump(out, open("mesures/resultats/mesure_vram.json", "w"), indent=2)
print(json.dumps(out, indent=2))
