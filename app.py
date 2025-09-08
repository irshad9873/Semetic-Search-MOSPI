# from flask import Flask, request, jsonify, render_template
# from sentence_transformers import SentenceTransformer
# import numpy as np
# import os, json

# # FAISS / Sklearn fallback
# USE_FAISS = True
# try:
#     import faiss
# except Exception:
#     USE_FAISS = False
#     from sklearn.neighbors import NearestNeighbors

# # ----------------- Load & Flatten Products -----------------
# PRODUCTS_FILE = os.path.join("products", "products.json")
# with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []
# PRODUCTS = []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     # Dataset
#     DATASETS.append({
#         "code": ds_name,
#         "name": ds_name,
#         "desc": ds_info.get("description", ""),
#         "type": "dataset"
#     })
#     PRODUCTS.append(DATASETS[-1])

#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind['name'],
#             "desc": ind.get("description", ""),
#             "type": "indicator",
#             "parent": ds_name
#         })
#         PRODUCTS.append(INDICATORS[-1])

#         for f in ind.get("filters", []):
#             if isinstance(f, dict):
#                 for fname, options in f.items():
#                     for opt in options:
#                         FILTERS.append({
#                             "code": f"{ind_code}_{fname}_{opt}".replace(" ", "_"),
#                             "name": fname,
#                             "desc": opt,
#                             "type": "filter",
#                             "parent": ind_code,
#                             "filter_name": fname,
#                             "option": opt
#                         })
#                         PRODUCTS.append(FILTERS[-1])

# print(f"Datasets={len(DATASETS)}, Indicators={len(INDICATORS)}, Filters={len(FILTERS)}")

# # ----------------- Embeddings + Index -----------------
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def build_index(products, model):
#     texts = [p["name"] + " " + str(p.get("desc", "")) for p in products]
#     if not texts:
#         raise ValueError("No products found to embed.")

#     embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

#     if len(embeddings.shape) == 1:
#         embeddings = embeddings.reshape(1, -1)

#     dim = embeddings.shape[1]

#     if USE_FAISS:
#         index = faiss.IndexFlatIP(dim)
#         index.add(embeddings)
#     else:
#         index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)

#     return embeddings, index

# PRODUCT_EMB, PRODUCT_INDEX = build_index(PRODUCTS, model)

# # ----------------- Helpers -----------------
# def semantic_search(query, candidates, top_k=5):
#     if not candidates:
#         return []

#     texts = [c["name"] + " " + str(c.get("desc", "")) for c in candidates]
#     emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
#     qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

#     results = []
#     if USE_FAISS:
#         index = faiss.IndexFlatIP(emb.shape[1])
#         index.add(emb)
#         sim, idx = index.search(qv, min(top_k, len(candidates)))
#         for r, i in enumerate(idx[0]):
#             c = candidates[i]
#             results.append({**c, "score": float(sim[0][r])})
#     else:
#         index = NearestNeighbors(n_neighbors=min(top_k, len(candidates)), metric="cosine").fit(emb)
#         distances, indices = index.kneighbors(qv)
#         for r, i in enumerate(indices[0]):
#             c = candidates[i]
#             score = 1.0 - float(distances[0][r])
#             results.append({**c, "score": score})

#     return results

# # ----------------- Flask App -----------------
# app = Flask(__name__, template_folder="templates")

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     q = request.json.get("query", "").strip()
#     if not q:
#         return jsonify({"error": "query required"}), 400

#     # Step 1: Dataset
#     dataset_results = semantic_search(q, DATASETS, top_k=3)
#     top_dataset = dataset_results[0] if dataset_results else None

#     # Step 2: Indicator
#     top_indicator = None
#     if top_dataset:
#         candidates = [i for i in INDICATORS if i["parent"] == top_dataset["code"]]
#         indicator_results = semantic_search(q, candidates, top_k=3)
#         top_indicator = indicator_results[0] if indicator_results else None

#     # Step 3: Filters (best per filter_name)
#     top_filters = {}
#     if top_indicator:
#         candidates = [f for f in FILTERS if f["parent"] == top_indicator["code"]]
#         filter_results = semantic_search(q, candidates, top_k=10)

#         # Group by filter_name
#         for f in filter_results:
#             fname = f.get("filter_name")
#             if not fname:
#                 continue
#             if fname not in top_filters or f["score"] > top_filters[fname]["score"]:
#                 top_filters[fname] = {
#                     "filter_name": fname,
#                     "option": f.get("option"),
#                     "score": f.get("score")
#                 }

#     return jsonify({
#         "query": q,
#         "dataset": top_dataset,
#         "indicator": top_indicator,
#         "filters": list(top_filters.values())  # multiple unique filters
#     })

# if __name__ == "__main__":
#     app.run(debug=True)



from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
import os, json

# FAISS / Sklearn fallback
USE_FAISS = True
try:
    import faiss
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

# Reranker
from llama_reranker import LlamaReranker

# ----------------- Load & Flatten Products -----------------
PRODUCTS_FILE = os.path.join("products", "products.json")
with open(PRODUCTS_FILE, "r", encoding="utf-8") as f:
    raw_products = json.load(f)

DATASETS, INDICATORS, FILTERS = [], [], []
PRODUCTS = []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     # Dataset
#     DATASETS.append({
#         "code": ds_name,
#         "name": ds_name,
#         "desc": ds_info.get("description", ""),
#         "type": "dataset"
#     })
#     PRODUCTS.append(DATASETS[-1])

#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind['name'],
#             "desc": ind.get("description", ""),
#             "type": "indicator",
#             "parent": ds_name
#         })
#         PRODUCTS.append(INDICATORS[-1])

#         for f in ind.get("filters", []):
#             if isinstance(f, dict):
#                 for fname, options in f.items():
#                     for opt in options:
#                         FILTERS.append({
#                             "code": f"{ind_code}_{fname}_{opt}".replace(" ", "_"),
#                             "name": fname,
#                             "desc": opt,
#                             "type": "filter",
#                             "parent": ind_code,
#                             "filter_name": fname,
#                             "option": opt
#                         })
#                         PRODUCTS.append(FILTERS[-1])

for ds_name, ds_info in raw_products.get("datasets", {}).items():
    # Dataset
    DATASETS.append({
        "code": ds_name,
        "name": ds_name,
        "desc": ds_info.get("description", ""),
        "type": "dataset"
    })
    PRODUCTS.append(DATASETS[-1])

    for ind in ds_info.get("indicators", []):
        ind_code = f"{ds_name}_{ind['name']}"
        INDICATORS.append({
            "code": ind_code,
            "name": ind['name'],
            "desc": ind.get("description", ""),
            "type": "indicator",
            "parent": ds_name
        })
        PRODUCTS.append(INDICATORS[-1])

        # ✅ filters optional bana diya
        if "filters" in ind and isinstance(ind["filters"], list):
            for f in ind["filters"]:
                if isinstance(f, dict):
                    for fname, options in f.items():
                        for opt in options:
                            FILTERS.append({
                                "code": f"{ind_code}_{fname}_{opt}".replace(" ", "_"),
                                "name": fname,
                                "desc": opt,
                                "type": "filter",
                                "parent": ind_code,
                                "filter_name": fname,
                                "option": opt
                            })
                            PRODUCTS.append(FILTERS[-1])


print(f"Datasets={len(DATASETS)}, Indicators={len(INDICATORS)}, Filters={len(FILTERS)}")

# ----------------- Embeddings + Index -----------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def build_index(products, model):
    texts = [p["name"] + " " + str(p.get("desc", "")) for p in products]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    if len(embeddings.shape) == 1:
        embeddings = embeddings.reshape(1, -1)
    dim = embeddings.shape[1]

    if USE_FAISS:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
    else:
        index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(embeddings)

    return embeddings, index

# Precompute all indexes (FAST reuse later)
DATASET_EMB, DATASET_INDEX = build_index(DATASETS, model)
INDICATOR_EMB, INDICATOR_INDEX = build_index(INDICATORS, model)
FILTER_EMB, FILTER_INDEX = build_index(FILTERS, model)

# ----------------- Reranker -----------------
RERANKER = None
try:
    RERANKER = LlamaReranker(
        model_path=os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
        products_file=PRODUCTS_FILE,
        ctx_size=2048,
        n_gpu_layers=0
    )
    print("✅ LLaMA reranker loaded.")
except Exception as e:
    print("⚠️ LLaMA reranker not loaded:", e)
    RERANKER = None

# ----------------- Helpers -----------------
def search_index(query, items, emb, index, top_k=5):
    if not items:
        return []
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    results = []
    if USE_FAISS:
        sim, idx = index.search(qv, min(top_k, len(items)))
        for r, i in enumerate(idx[0]):
            c = items[i]
            results.append({**c, "score": float(sim[0][r])})
    else:
        distances, indices = index.kneighbors(qv)
        for r, i in enumerate(indices[0]):
            c = items[i]
            score = 1.0 - float(distances[0][r])
            results.append({**c, "score": score})
    return results

def rerank(query, shortlist, blend_weight_llm=0.6):
    if RERANKER and shortlist:
        try:
            return RERANKER.rerank(query, shortlist, blend_weight_llm=blend_weight_llm)
        except Exception as e:
            print("⚠️ Rerank failed:", e)
    return sorted(shortlist, key=lambda x: x["score"], reverse=True)

# ----------------- Flask App -----------------
app = Flask(__name__, template_folder="templates")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    q = request.json.get("query", "").strip()
    if not q:
        return jsonify({"error": "query required"}), 400

    # -------- Dataset --------
    dataset_results = search_index(q, DATASETS, DATASET_EMB, DATASET_INDEX, top_k=3)
    dataset_reranked = rerank(q, dataset_results)   # llama only here
    top_dataset = dataset_reranked[0] if dataset_reranked else None

    # -------- Indicator --------
    top_indicator = None
    if top_dataset:
        candidates = [i for i in INDICATORS if i["parent"] == top_dataset["code"]]
        if candidates:
            texts = [c["name"] + " " + str(c.get("desc", "")) for c in candidates]
            candidate_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            if USE_FAISS:
                dim = candidate_emb.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(candidate_emb)
            else:
                index = NearestNeighbors(n_neighbors=min(3, len(candidates)), metric="cosine").fit(candidate_emb)

            indicator_results = search_index(q, candidates, candidate_emb, index, top_k=3)
            indicator_reranked = rerank(q, indicator_results)   # llama only here
            top_indicator = indicator_reranked[0] if indicator_reranked else None

    # -------- Filters --------
    top_filters = []
    if top_indicator:
        candidates = [f for f in FILTERS if f["parent"] == top_indicator["code"]]
        filter_groups = {}
        for f in candidates:
            filter_groups.setdefault(f["filter_name"], []).append(f)

        # pick best option per filter_name
        for fname, group in filter_groups.items():
            texts = [f["name"] + " " + str(f.get("desc", "")) for f in group]
            group_emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            if USE_FAISS:
                dim = group_emb.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(group_emb)
            else:
                index = NearestNeighbors(n_neighbors=1, metric="cosine").fit(group_emb)

            # SIM_THRESHOLD = 0.15  # only show filters with score >= 0.3

            # group_results = search_index(q, group, group_emb, index, top_k=1)
            # if group_results:
            #     best = group_results[0]
            #     if best.get("score", 0) >= SIM_THRESHOLD:
            #         top_filters.append({
            #             "filter_name": best.get("filter_name", "UNKNOWN"),
            #             "option": best.get("option", ""),
            #             "score": best.get("score", 0)
            #         })

            group_results = search_index(q, group, group_emb, index, top_k=1)
            if group_results:
                best = group_results[0]
                top_filters.append({
                    "filter_name": best.get("filter_name", "UNKNOWN"),
                    "option": best.get("option", ""),
                    "score": best.get("score", 0)
                })



    return jsonify({
        "query": q,
        "dataset": top_dataset,
        "indicator": top_indicator,
        "filters": top_filters
    })


if __name__ == "__main__":
    app.run(debug=True)
