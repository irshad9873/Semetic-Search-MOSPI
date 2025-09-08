# # from flask import Flask, request, jsonify
# # from sentence_transformers import SentenceTransformer
# # import numpy as np
# # import os, json
# # from flask import Flask, request, jsonify, render_template

# # USE_FAISS = True
# # try:
# #     import faiss  # if available
# # except Exception:
# #     USE_FAISS = False
# #     from sklearn.neighbors import NearestNeighbors

# # #  Load products from JSON file
# # PRODUCTS_FILE = os.path.join("products", "products.json")
# # with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
# #     PRODUCTS = json.load(f)

# # app = Flask(__name__)

# # # Load model (downloads on first run if not cached)
# # model = SentenceTransformer("all-MiniLM-L6-v2")
# # texts = [f"{p['name']} - {p['desc']}" for p in PRODUCTS]
# # emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# # # Build index
# # if USE_FAISS:
# #     d = emb.shape[1]
# #     index = faiss.IndexFlatIP(d)  # cosine (because normalized)
# #     index.add(emb)
# # else:
# #     nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(emb)

# # # ---------------------------
# # #  Home route (fix for 404)
# # # ---------------------------
# # # @app.route("/")
# # # def home():
# # #     return "Flask Product Prediction API is running! Use POST /predict_product with JSON {\"query\": \"your text\"}."

# # @app.route("/")
# # def home():
# #     return render_template("index.html")


# # @app.route("/predict_product", methods=["POST"])
# # def predict_product():
# #     q = request.json.get("query", "").strip()
# #     print(q)
# #     if not q:
# #         return jsonify({"error": "query required"}), 400

# #     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

# #     if USE_FAISS:
# #         sim, idx = index.search(qv, 3)
# #         scores = sim[0]; ids = idx[0]
# #         results = [{
# #             "rank": r+1,
# #             "product_code": PRODUCTS[i]["code"],
# #             "product_name": PRODUCTS[i]["name"],
# #             "score": float(scores[r])
# #         } for r,i in enumerate(ids)]
# #     else:
# #         distances, indices = nn.kneighbors(qv, n_neighbors=3, return_distance=True)
# #         results = []
# #         for r, i in enumerate(indices[0]):
# #             sim = 1.0 - float(distances[0][r])
# #             p = PRODUCTS[i]
# #             results.append({
# #                 "rank": r+1,
# #                 "product_code": p["code"],
# #                 "product_name": p["name"],
# #                 "score": sim
# #             })

# #     return jsonify({"query": q, "predicted_products": results})

# # if __name__ == "__main__":
# #     app.run(debug=True)


# # app.py
# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # Load products
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     PRODUCTS = json.load(f)

# app = Flask(__name__, template_folder="templates")

# # SBERT embeddings + index
# model = SentenceTransformer("all-MiniLM-L6-v2")
# texts = [f"{p['name']} - {p['desc']}" for p in PRODUCTS]
# emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
# if USE_FAISS:
#     d = emb.shape[1]
#     index = faiss.IndexFlatIP(d)
#     index.add(emb)
# else:
#     nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(emb)

# # Try to load LLaMA reranker (optional)
# USE_LLAMA = False
# RERANKER = None
# try:
#     from llama_reranker import LlamaReranker
#     RERANKER = LlamaReranker(
#         model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file=PRODUCTS_FILE,
#         ctx_size=2048,
#         n_gpu_layers=0,
#     )
#     USE_LLAMA = True
#     print("LLaMA reranker loaded.")
# except Exception as e:
#     print("LLaMA reranker not loaded:", e)
#     USE_LLAMA = False

# @app.route("/")
# def home():
#     return render_template("index.html")

# # old endpoint (embedding only)
# @app.route("/predict_product", methods=["POST"])
# def predict_product():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
#     if USE_FAISS:
#         sim, idx = index.search(qv, 3)
#         scores = sim[0]; ids = idx[0]
#         results = [{
#             "rank": r+1,
#             "product_code": PRODUCTS[i]["code"],
#             "product_name": PRODUCTS[i]["name"],
#             "score": float(scores[r])
#         } for r,i in enumerate(ids)]
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=3, return_distance=True)
#         results = []
#         for r, i in enumerate(indices[0]):
#             sim = 1.0 - float(distances[0][r])
#             p = PRODUCTS[i]
#             results.append({
#                 "rank": r+1,
#                 "product_code": p["code"],
#                 "product_name": p["name"],
#                 "score": sim
#             })

#     return jsonify({"query": q, "predicted_products": results})

# # new endpoint (embedding shortlist -> LLaMA rerank)
# @app.route("/predict_product_llama", methods=["POST"])
# def predict_product_llama():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400
#     if not USE_LLAMA or RERANKER is None:
#         return jsonify({"error": "LLAMA reranker not available"}), 503

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

#     # get top-5 shortlist from embeddings
#     K = 5
#     shortlist = []
#     if USE_FAISS:
#         sim, idx = index.search(qv, K)
#         for rank, (s, i) in enumerate(zip(sim[0], idx[0]), start=1):
#             p = PRODUCTS[i]
#             shortlist.append({
#                 "rank": rank,
#                 "product_code": p["code"],
#                 "product_name": p["name"],
#                 "desc": p["desc"],
#                 "score": float(s),
#                 "code": p["code"]
#             })
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=K, return_distance=True)
#         for rank, i in enumerate(indices[0], start=1):
#             simi = 1.0 - float(distances[0][rank-1])
#             p = PRODUCTS[i]
#             shortlist.append({
#                 "rank": rank,
#                 "product_code": p["code"],
#                 "product_name": p["name"],
#                 "desc": p["desc"],
#                 "score": simi,
#                 "code": p["code"]
#             })

#     reranked = RERANKER.rerank(q, shortlist, blend_weight_llm=0.6)

#     out = []
#     for r, p in enumerate(reranked, start=1):
#         out.append({
#             "rank": r,
#             "product_code": p["product_code"],
#             "product_name": p["product_name"],
#             "embed_score": p["score"],
#             "llm_choice": p["llm_choice"],
#             "llm_confidence": p["llm_confidence"],
#             "final_score": p["final_score"],
#             "reason": p.get("reason", "")
#         })

#     return jsonify({"query": q, "predicted_products": out})

# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # Load products
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     PRODUCTS = json.load(f)

# app = Flask(__name__, template_folder="templates")

# # SBERT embeddings + index
# model = SentenceTransformer("all-MiniLM-L6-v2")
# texts = [f"{p.get('name','UNKNOWN')} - {p.get('desc','')}" for p in PRODUCTS]
# emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# if USE_FAISS:
#     d = emb.shape[1]
#     index = faiss.IndexFlatIP(d)
#     index.add(emb)
# else:
#     nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(emb)

# # Try to load LLaMA reranker (optional)
# USE_LLAMA = False
# RERANKER = None
# try:
#     from llama_reranker import LlamaReranker
#     RERANKER = LlamaReranker(
#         model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file=PRODUCTS_FILE,
#         ctx_size=2048,
#         n_gpu_layers=0,
#     )
#     USE_LLAMA = True
#     print("LLaMA reranker loaded.")
# except Exception as e:
#     print("LLaMA reranker not loaded:", e)
#     USE_LLAMA = False

# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict_product", methods=["POST"])
# def predict_product():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

#     results = []
#     if USE_FAISS:
#         sim, idx = index.search(qv, 3)
#         for r, i in enumerate(idx[0], start=1):
#             p = PRODUCTS[i]
#             results.append({
#                 "rank": r,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "score": float(sim[0][r-1])
#             })
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=3, return_distance=True)
#         for r, i in enumerate(indices[0], start=1):
#             p = PRODUCTS[i]
#             sim_score = 1.0 - float(distances[0][r-1])
#             results.append({
#                 "rank": r,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "score": sim_score
#             })

#     return jsonify({"query": q, "predicted_products": results})


# @app.route("/predict_product_llama", methods=["POST"])
# def predict_product_llama():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400
#     if not USE_LLAMA or RERANKER is None:
#         return jsonify({"error": "LLAMA reranker not available"}), 503

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

#     # get top-5 shortlist from embeddings
#     K = 5
#     shortlist = []
#     if USE_FAISS:
#         sim, idx = index.search(qv, K)
#         for rank, (s, i) in enumerate(zip(sim[0], idx[0]), start=1):
#             p = PRODUCTS[i]
#             shortlist.append({
#                 "rank": rank,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "name": p.get("name", "UNKNOWN"),   # required by LLaMA
#                 "desc": p.get("desc", ""),
#                 "score": float(s),
#                 "code": p.get("code", "UNKNOWN")
#             })
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=K, return_distance=True)
#         for rank, i in enumerate(indices[0], start=1):
#             p = PRODUCTS[i]
#             sim_score = 1.0 - float(distances[0][rank-1])
#             shortlist.append({
#                 "rank": rank,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "name": p.get("name", "UNKNOWN"),
#                 "desc": p.get("desc", ""),
#                 "score": sim_score,
#                 "code": p.get("code", "UNKNOWN")
#             })

#     reranked = RERANKER.rerank(q, shortlist, blend_weight_llm=0.6)

#     out = []
#     for r, p in enumerate(reranked, start=1):
#         out.append({
#             "rank": r,
#             "product_code": p["code"],
#             "product_name": p["name"],
#             "embed_score": p["score"],
#             "llm_choice": p["llm_choice"],
#             "llm_confidence": p["llm_confidence"],
#             "final_score": p["final_score"],
#             "reason": p.get("reason", "")
#         })

#     return jsonify({"query": q, "predicted_products": out})


# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # Load products JSON
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     raw_products = json.load(f)

# # Flatten datasets + indicators
# PRODUCTS = []
# datasets = raw_products.get("datasets", {})
# for ds_name, ds_info in datasets.items():
#     # Dataset itself
#     PRODUCTS.append({
#         "code": ds_name,
#         "name": ds_name,
#         "desc": ds_info.get("description", "")
#     })
#     # Each indicator
#     for ind in ds_info.get("indicators", []):
#         PRODUCTS.append({
#             "code": f"{ds_name}_{ind['name']}",
#             "name": ind['name'],
#             "desc": ind.get("description", "")
#         })

# # Flask app
# app = Flask(__name__, template_folder="templates")

# # SBERT embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")
# texts = [f"{p['name']} - {p['desc']}" for p in PRODUCTS]
# emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

# if USE_FAISS:
#     d = emb.shape[1]
#     index = faiss.IndexFlatIP(d)
#     index.add(emb)
# else:
#     nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(emb)

# # Try to load LLaMA reranker (optional)
# USE_LLAMA = False
# RERANKER = None
# try:
#     from llama_reranker import LlamaReranker
#     RERANKER = LlamaReranker(
#         model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file=PRODUCTS_FILE,
#         ctx_size=2048,
#         n_gpu_layers=0,
#     )
#     USE_LLAMA = True
#     print("LLaMA reranker loaded.")
# except Exception as e:
#     print("LLaMA reranker not loaded:", e)
#     USE_LLAMA = False


# @app.route("/")
# def home():
#     return render_template("index.html")


# @app.route("/predict_product", methods=["POST"])
# def predict_product():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)

#     results = []
#     K = 5
#     if USE_FAISS:
#         sim, idx = index.search(qv, K)
#         for r, i in enumerate(idx[0], start=1):
#             p = PRODUCTS[i]
#             results.append({
#                 "rank": r,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "score": float(sim[0][r-1])
#             })
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=K, return_distance=True)
#         for r, i in enumerate(indices[0], start=1):
#             p = PRODUCTS[i]
#             sim_score = 1.0 - float(distances[0][r-1])
#             results.append({
#                 "rank": r,
#                 "product_code": p.get("code", "UNKNOWN"),
#                 "product_name": p.get("name", "UNKNOWN"),
#                 "score": sim_score
#             })

#     return jsonify({"query": q, "predicted_products": results})


# @app.route("/predict_product_llama", methods=["POST"])
# def predict_product_llama():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     qv = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)
#     K = 5
#     shortlist = []

#     # Get top-K candidates from embeddings
#     if USE_FAISS:
#         sim, idx = index.search(qv, K)
#         for rank, (s, i) in enumerate(zip(sim[0], idx[0]), start=1):
#             p = PRODUCTS[i]
#             shortlist.append({
#                 "rank": rank,
#                 "code": p.get("code", "UNKNOWN"),
#                 "name": p.get("name", "UNKNOWN"),
#                 "desc": p.get("desc", ""),
#                 "score": float(s)
#             })
#     else:
#         distances, indices = nn.kneighbors(qv, n_neighbors=K, return_distance=True)
#         for rank, i in enumerate(indices[0], start=1):
#             p = PRODUCTS[i]
#             sim_score = 1.0 - float(distances[0][rank-1])
#             shortlist.append({
#                 "rank": rank,
#                 "code": p.get("code", "UNKNOWN"),
#                 "name": p.get("name", "UNKNOWN"),
#                 "desc": p.get("desc", ""),
#                 "score": sim_score
#             })

#     if USE_LLAMA and RERANKER:
#         reranked = RERANKER.rerank(q, shortlist, blend_weight_llm=0.6)
#     else:
#         # Fallback: just sort by embedding score
#         reranked = sorted(shortlist, key=lambda x: x["score"], reverse=True)
#         for r, p in enumerate(reranked, start=1):
#             p["rank"] = r
#             p["llm_choice"] = False
#             p["llm_confidence"] = 0.0
#             p["final_score"] = p["score"]
#             p["reason"] = ""

#     return jsonify({"query": q, "predicted_products": reranked})


# if __name__ == "__main__":
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # Load products JSON
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     raw_products = json.load(f)

# # Flatten datasets and indicators
# DATASETS = []
# INDICATORS = {}  # mapping: dataset_code -> list of indicators
# datasets = raw_products.get("datasets", {})
# for ds_name, ds_info in datasets.items():
#     ds_code = ds_name.strip()
#     DATASETS.append({
#         "code": ds_code,
#         "name": ds_name.strip(),
#         "desc": ds_info.get("description","").strip()
#     })
#     INDICATORS[ds_code] = []
#     for ind in ds_info.get("indicators", []):
#         INDICATORS[ds_code].append({
#             "code": f"{ds_code}_{ind['name']}".strip(),
#             "name": ind['name'].strip(),
#             "desc": ind.get("description","").strip()
#         })

# # Initialize Flask app
# app = Flask(__name__, template_folder="templates")

# # SBERT model for embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Dataset embeddings
# dataset_texts = [f"{d['name']} - {d['desc']}" for d in DATASETS]
# dataset_emb = model.encode(dataset_texts, convert_to_numpy=True, normalize_embeddings=True)
# if USE_FAISS:
#     d_dim = dataset_emb.shape[1]
#     dataset_index = faiss.IndexFlatIP(d_dim)
#     dataset_index.add(dataset_emb)
# else:
#     dataset_nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(dataset_emb)

# # LLaMA reranker (optional)
# USE_LLAMA = False
# RERANKER = None
# try:
#     from llama_reranker import LlamaReranker
#     RERANKER = LlamaReranker(
#         model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file=PRODUCTS_FILE,
#         ctx_size=2048,
#         n_gpu_layers=0
#     )
#     USE_LLAMA = True
#     print("LLaMA reranker loaded.")
# except Exception as e:
#     print("LLaMA reranker not loaded:", e)
#     USE_LLAMA = False

# @app.route("/")
# def home():
#     return render_template("index.html")

# # --- Helper functions ---
# def semantic_search(query, candidates, top_k=5):
#     texts = [f"{c['name']} - {c['desc']}" for c in candidates]
#     qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#     if USE_FAISS:
#         sim, idx = dataset_index.search(qv, min(top_k, len(candidates)))
#         results = []
#         for r, i in enumerate(idx[0]):
#             c = candidates[i]
#             results.append({
#                 "code": c.get("code","UNKNOWN"),
#                 "name": c.get("name","UNKNOWN"),
#                 "desc": c.get("desc",""),
#                 "score": float(sim[0][r])
#             })
#         return results
#     else:
#         nn = NearestNeighbors(n_neighbors=top_k, metric="cosine").fit(np.array([model.encode([c['name'] + " " + c['desc']], convert_to_numpy=True) for c in candidates]).squeeze())
#         distances, indices = nn.kneighbors(qv)
#         results = []
#         for r, i in enumerate(indices[0]):
#             c = candidates[i]
#             results.append({
#                 "code": c.get("code","UNKNOWN"),
#                 "name": c.get("name","UNKNOWN"),
#                 "desc": c.get("desc",""),
#                 "score": 1.0 - float(distances[0][r])
#             })
#         return results

# def rerank(query, shortlist, blend_weight_llm=0.6):
#     if USE_LLAMA and RERANKER:
#         try:
#             return RERANKER.rerank(query, shortlist, blend_weight_llm=blend_weight_llm)
#         except Exception as e:
#             print("LLaMA rerank failed:", e)
#     # fallback: sort by semantic score
#     return sorted(shortlist, key=lambda x: x["score"], reverse=True)

# # --- API endpoint ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query","").strip()
#     if not q:
#         return jsonify({"error":"query required"}), 400

#     # Step 1: Predict dataset
#     dataset_candidates = semantic_search(q, DATASETS, top_k=3)
#     dataset_reranked = rerank(q, dataset_candidates, blend_weight_llm=0.6)
#     top_dataset = dataset_reranked[0]

#     # Step 2: Predict indicator
#     indicator_list = INDICATORS.get(top_dataset["code"], [])
#     if indicator_list:
#         indicator_candidates = semantic_search(q, indicator_list, top_k=5)
#         indicator_reranked = rerank(q, indicator_candidates, blend_weight_llm=0.6)
#         top_indicator = indicator_reranked[0]
#     else:
#         top_indicator = None

#     return jsonify({
#         "query": q,
#         "dataset": top_dataset,
#         "indicator": top_indicator
#     })

# if __name__ == "__main__":
#     app.run(debug=True)



# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # Load products JSON
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     raw_products = json.load(f)

# # Flatten datasets and indicators
# DATASETS = []
# INDICATORS = {}  # mapping: dataset_code -> list of indicators
# datasets = raw_products.get("datasets", {})
# for ds_name, ds_info in datasets.items():
#     ds_code = ds_name.strip()
#     DATASETS.append({
#         "code": ds_code,
#         "name": ds_name.strip(),
#         "desc": ds_info.get("description","").strip()
#     })
#     INDICATORS[ds_code] = []
#     for ind in ds_info.get("indicators", []):
#         INDICATORS[ds_code].append({
#             "code": f"{ds_code}_{ind['name']}".strip(),
#             "name": ind['name'].strip(),
#             "desc": ind.get("description","").strip()
#         })

# # Initialize Flask app
# app = Flask(__name__, template_folder="templates")

# # SBERT model for embeddings
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Dataset embeddings
# dataset_texts = [f"{d['name']} - {d['desc']}" for d in DATASETS]
# dataset_emb = model.encode(dataset_texts, convert_to_numpy=True, normalize_embeddings=True)
# if USE_FAISS:
#     d_dim = dataset_emb.shape[1]
#     dataset_index = faiss.IndexFlatIP(d_dim)
#     dataset_index.add(dataset_emb)
# else:
#     dataset_nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(dataset_emb)

# # LLaMA reranker (optional)
# USE_LLAMA = False
# RERANKER = None
# try:
#     from llama_reranker import LlamaReranker
#     RERANKER = LlamaReranker(
#         model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file=PRODUCTS_FILE,
#         ctx_size=2048,
#         n_gpu_layers=0
#     )
#     USE_LLAMA = True
#     print("LLaMA reranker loaded.")
# except Exception as e:
#     print("LLaMA reranker not loaded:", e)
#     USE_LLAMA = False

# @app.route("/")
# def home():
#     return render_template("index.html")

# # --- Helper functions ---
# def semantic_search(query, candidates, top_k=5):
#     qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
#     if USE_FAISS:
#         sim, idx = dataset_index.search(qv, min(top_k, len(candidates)))
#         results = []
#         for r, i in enumerate(idx[0]):
#             c = candidates[i]
#             results.append({
#                 "code": c.get("code","UNKNOWN"),
#                 "name": c.get("name","UNKNOWN"),
#                 "desc": c.get("desc",""),
#                 "score": float(sim[0][r])
#             })
#         return results
#     else:
#         embeddings = np.array([model.encode([c['name'] + " " + c['desc']], convert_to_numpy=True).squeeze() for c in candidates])
#         nn = NearestNeighbors(n_neighbors=min(top_k, len(candidates)), metric="cosine").fit(embeddings)
#         distances, indices = nn.kneighbors(qv)
#         results = []
#         for r, i in enumerate(indices[0]):
#             c = candidates[i]
#             results.append({
#                 "code": c.get("code","UNKNOWN"),
#                 "name": c.get("name","UNKNOWN"),
#                 "desc": c.get("desc",""),
#                 "score": 1.0 - float(distances[0][r])
#             })
#         return results

# def rerank(query, shortlist, blend_weight_llm=0.6):
#     if USE_LLAMA and RERANKER:
#         try:
#             reranked = RERANKER.rerank(query, shortlist, blend_weight_llm=blend_weight_llm)
#             print("LLaMA rerank result:", reranked)
#             return reranked
#         except Exception as e:
#             print("LLaMA rerank failed:", e)
#     # fallback: sort by semantic score
#     return sorted(shortlist, key=lambda x: x["score"], reverse=True)

# # --- API endpoint ---
# @app.route("/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query","").strip()
#     if not q:
#         return jsonify({"error":"query required"}), 400

#     # Step 1: Predict dataset
#     dataset_candidates = semantic_search(q, DATASETS, top_k=3)
#     print('**************************************************************',dataset_candidates)
#     dataset_reranked = rerank(q, dataset_candidates, blend_weight_llm=0.6)
#     top_dataset = dataset_reranked[0] if dataset_reranked else None

#     # Step 2: Predict indicator
#     top_indicator = None
#     if top_dataset:
#         indicator_list = INDICATORS.get(top_dataset["code"], [])
#         if indicator_list:
#             indicator_candidates = semantic_search(q, indicator_list, top_k=5)
#             indicator_reranked = rerank(q, indicator_candidates, blend_weight_llm=0.6)
#             top_indicator = indicator_reranked[0] if indicator_reranked else None

#     return jsonify({
#         "query": q,
#         "dataset": top_dataset,
#         "indicator": top_indicator
#     })

# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
import os, json

USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# Load products JSON
PRODUCTS_FILE = os.path.join("products", "products.json")
with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    raw_products = json.load(f)

# Flatten datasets and indicators
DATASETS = []
INDICATORS = {}  # mapping: dataset_code -> list of indicators
FILTERS = []     # flat list of filters with indicator reference
datasets = raw_products.get("datasets", {})
for ds_name, ds_info in datasets.items():
    ds_code = ds_name.strip()
    DATASETS.append({
        "code": ds_code,
        "name": ds_name.strip(),
        "desc": ds_info.get("description","").strip()
    })
    INDICATORS[ds_code] = []
    for ind in ds_info.get("indicators", []):
        ind_code = f"{ds_code}_{ind['name']}".strip()
        INDICATORS[ds_code].append({
            "code": ind_code,
            "name": ind['name'].strip(),
            "desc": ind.get("description","").strip()
        })
        # Flatten filters
        for f in ind.get("filters", []):
            for filter_name, options in f.items():
                FILTERS.append({
                    "indicator_code": ind_code,
                    "filter_name": filter_name,
                    "options": options
                })

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# SBERT model for embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Dataset embeddings (built globally, as dataset list is small)
dataset_texts = [f"{d['name']} - {d['desc']}" for d in DATASETS]
dataset_emb = model.encode(dataset_texts, convert_to_numpy=True, normalize_embeddings=True)
if USE_FAISS:
    d_dim = dataset_emb.shape[1]
    dataset_index = faiss.IndexFlatIP(d_dim)
    dataset_index.add(dataset_emb)
else:
    dataset_nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(dataset_emb)

# LLaMA reranker (optional)
USE_LLAMA = False
RERANKER = None
try:
    from llama_reranker import LlamaReranker
    RERANKER = LlamaReranker(
        model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
        products_file=PRODUCTS_FILE,
        ctx_size=2048,
        n_gpu_layers=0
    )
    USE_LLAMA = True
    print("LLaMA reranker loaded.")
except Exception as e:
    print("LLaMA reranker not loaded:", e)
    USE_LLAMA = False

# --- Helper functions ---
def semantic_search(query, candidates, top_k=5, candidate_type="dataset"):
    if not candidates:
        return []
    
    # Build embeddings dynamically for this subset
    texts = []
    for c in candidates:
        if candidate_type == "filter":
            texts.append(f"{c['filter_name']} - options: {', '.join(map(str, c['options']))}")
        else:
            texts.append(f"{c['name']} - {c.get('desc','')}")
    
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    if USE_FAISS:
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
    else:
        index = NearestNeighbors(n_neighbors=min(top_k, len(candidates)), metric="cosine").fit(emb)

    # Encode query
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    results = []
    if USE_FAISS:
        sim, idx = index.search(qv, min(top_k, len(candidates)))
        for r, i in enumerate(idx[0]):
            c = candidates[i]
            score = float(sim[0][r])
            results.append({**c, "score": score})
    else:
        distances, indices = index.kneighbors(qv)
        for r, i in enumerate(indices[0]):
            c = candidates[i]
            score = 1.0 - float(distances[0][r])
            results.append({**c, "score": score})

    return results

def rerank(query, shortlist, blend_weight_llm=0.6):
    if USE_LLAMA and RERANKER:
        try:
            reranked = RERANKER.rerank(query, shortlist, blend_weight_llm=blend_weight_llm)
            return reranked
        except Exception as e:
            print("LLaMA rerank failed:", e)
    # fallback: sort by semantic score
    return sorted(shortlist, key=lambda x: x["score"], reverse=True)

# --- Flask routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    q = request.json.get("query","").strip()
    if not q:
        return jsonify({"error":"query required"}), 400

    # Step 1: Predict dataset
    dataset_candidates = semantic_search(q, DATASETS, top_k=3, candidate_type="dataset")
    dataset_reranked = rerank(q, dataset_candidates)
    top_dataset = dataset_reranked[0] if dataset_reranked else None

    # Step 2: Predict indicator
    top_indicator = None
    if top_dataset:
        indicator_list = INDICATORS.get(top_dataset["code"], [])
        if indicator_list:
            indicator_results = semantic_search(q, indicator_list, top_k=5, candidate_type="indicator")
            indicator_results = rerank(q, indicator_results)
            top_indicator = indicator_results[0] if indicator_results else None

    # Step 3: Predict filters
    top_filters = []
    if top_indicator:
        relevant_filters = [f for f in FILTERS if f["indicator_code"] == top_indicator["code"]]
        if relevant_filters:
            filter_results = semantic_search(q, relevant_filters, top_k=5, candidate_type="filter")
            top_filters = rerank(q, filter_results)

    return jsonify({
        "query": q,
        "dataset": top_dataset,
        "indicator": top_indicator,
        "filters": top_filters
    })

if __name__ == "__main__":
    app.run(debug=True)
