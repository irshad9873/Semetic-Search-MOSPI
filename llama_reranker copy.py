# # llama_reranker.py
# import os, json, re
# from typing import List, Dict, Any, Tuple

# try:
#     from llama_cpp import Llama
# except Exception as e:
#     raise RuntimeError(
#         "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
#     ) from e


# class LlamaReranker:
#     def __init__(
#         self,
#         model_path: str = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file: str = os.path.join("products", "products.json"),
#         ctx_size: int = 2048,
#         n_gpu_layers: int = 0,
#         n_threads: int = None,
#         temperature: float = 0.1,
#         top_p: float = 0.9,
#         max_tokens: int = 128,
#     ):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(
#                 f"GGUF model not found at {model_path}. Place quantized 7B model there."
#             )
#         if not os.path.exists(products_file):
#             raise FileNotFoundError(f"Products file not found: {products_file}")

#         self.products: List[Dict[str, Any]] = json.load(open(products_file, "r", encoding="utf-8"))
#         self.llm = Llama(
#             model_path=model_path,
#             n_ctx=ctx_size,
#             n_gpu_layers=n_gpu_layers,
#             n_threads=n_threads or os.cpu_count(),
#             verbose=False,
#         )
#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens

#     def _build_choice_block(self, candidates: List[Dict[str, Any]]) -> str:
#         lines = []
#         for i, p in enumerate(candidates, start=1):
#             lines.append(
#                 f"{i}. CODE={p['code']}\n"
#                 f"   NAME={p['name']}\n"
#                 f"   DESC={p['desc']}\n"
#             )
#         return "\n".join(lines)

#     def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
#         choices = self._build_choice_block(candidates)
#         system = (
#             "You are a product router. Given a user query and a small catalog of products, "
#             "pick the SINGLE best matching product CODE. Return strictly in JSON."
#         )
#         user = f"""Query: {query}

# Catalog (shortlist):
# {choices}

# Rules:
# - Consider sector/measure/actor words in query (e.g., employment, CPI, start-ups, crime, labour, PLFS, MSME, banks, SHG).
# - Choose exactly ONE product code from the shortlist that best answers the query.
# - Output strict JSON: {{"code": "<code>", "confidence": <0..1>, "reason": "<short reason>"}}"""

#         prompt = f"{system}\n\n{user}\n"
#         return prompt

#     def _parse_json(self, text: str) -> Tuple[str, float, str]:
#         m = re.search(r"\{.*\}", text, flags=re.DOTALL)
#         if not m:
#             return "", 0.0, "parse_failed"
#         block = m.group(0)
#         try:
#             obj = json.loads(block)
#             code = str(obj.get("code", "")).strip()
#             conf = float(obj.get("confidence", 0.0))
#             reason = str(obj.get("reason", "")).strip()
#             if not (0.0 <= conf <= 1.0):
#                 conf = max(0.0, min(1.0, conf))
#             return code, conf, reason
#         except Exception:
#             code_m = re.search(r'"code"\s*:\s*"([^"]+)"', block)
#             conf_m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', block)
#             code = code_m.group(1) if code_m else ""
#             conf = float(conf_m.group(1)) if conf_m else 0.0
#             return code, conf, "regex_fallback"

#     def rerank(
#         self,
#         query: str,
#         shortlist: List[Dict[str, Any]],
#         blend_weight_llm: float = 0.6,
#     ) -> List[Dict[str, Any]]:
#         if not shortlist:
#             return []

#         prompt = self._build_prompt(query, shortlist)
#         out = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, echo=False)
#         text = out["choices"][0]["text"]
#         code, conf, reason = self._parse_json(text)

#         results = []
#         for p in shortlist:
#             embed_sim = float(p.get("score", 0.0))
#             llm_boost = conf if p["code"].lower() == code.lower() else 0.0
#             final_score = blend_weight_llm * llm_boost + (1.0 - blend_weight_llm) * embed_sim
#             results.append({
#                 **p,
#                 "llm_choice": (p["code"].lower() == code.lower()),
#                 "llm_confidence": conf if p["code"].lower() == code.lower() else 0.0,
#                 "reason": reason if p["code"].lower() == code.lower() else "",
#                 "final_score": final_score,
#             })

#         results.sort(key=lambda x: x["final_score"], reverse=True)
#         return results






# import os
# import json
# import re
# from typing import List, Dict, Any, Tuple

# try:
#     from llama_cpp import Llama
# except ImportError:
#     raise RuntimeError(
#         "llama-cpp-python is not installed. Install with: pip install llama-cpp-python"
#     )

# class LlamaReranker:
#     def __init__(
#         self,
#         model_path: str = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
#         products_file: str = os.path.join("products", "products.json"),
#         ctx_size: int = 2048,
#         n_gpu_layers: int = 0,
#         n_threads: int = None,
#         temperature: float = 0.1,
#         top_p: float = 0.9,
#         max_tokens: int = 128,
#     ):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(
#                 f"GGUF model not found at {model_path}. Place quantized 7B model there."
#             )
#         if not os.path.exists(products_file):
#             raise FileNotFoundError(f"Products file not found: {products_file}")

#         # Load products JSON
#         with open(products_file, "r", encoding="utf-8") as f:
#             self.products: List[Dict[str, Any]] = json.load(f)

#         # Initialize LLaMA model
#         self.llm = Llama(
#             model_path=model_path,
#             n_ctx=ctx_size,
#             n_gpu_layers=n_gpu_layers,
#             n_threads=n_threads or os.cpu_count(),
#             verbose=False,
#         )

#         self.temperature = temperature
#         self.top_p = top_p
#         self.max_tokens = max_tokens

#     def _build_choice_block(self, candidates: List[Dict[str, Any]]) -> str:
#         """Builds the product shortlist block for the prompt"""
#         lines = []
#         for i, p in enumerate(candidates, start=1):
#             code = p.get("code", "UNKNOWN")
#             name = p.get("name", "UNKNOWN")
#             desc = p.get("desc", "No description available")
#             lines.append(f"{i}. CODE={code}\n   NAME={name}\n   DESC={desc}\n")
#         return "\n".join(lines)

#     def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
#         """Construct the full system + user prompt for LLaMA"""
#         choices = self._build_choice_block(candidates)
#         system = (
#             "You are a product router. Given a user query and a small catalog of products, "
#             "pick the SINGLE best matching product CODE. Return strictly in JSON."
#         )
#         user = f"""Query: {query}

# Catalog (shortlist):
# {choices}

# Rules:
# - Consider sector/measure/actor words in query (e.g., employment, CPI, start-ups, crime, labour, PLFS, MSME, banks, SHG).
# - Choose exactly ONE product code from the shortlist that best answers the query.
# - Output strict JSON: {{"code": "<code>", "confidence": <0..1>, "reason": "<short reason>"}}"""

#         return f"{system}\n\n{user}\n"

#     def _parse_json(self, text: str) -> Tuple[str, float, str]:
#         """Parse the JSON returned by the model"""
#         m = re.search(r"\{.*\}", text, flags=re.DOTALL)
#         if not m:
#             return "", 0.0, "parse_failed"
#         block = m.group(0)
#         try:
#             obj = json.loads(block)
#             code = str(obj.get("code", "")).strip()
#             conf = float(obj.get("confidence", 0.0))
#             reason = str(obj.get("reason", "")).strip()
#             if not (0.0 <= conf <= 1.0):
#                 conf = max(0.0, min(1.0, conf))
#             return code, conf, reason
#         except Exception:
#             # fallback regex parsing
#             code_m = re.search(r'"code"\s*:\s*"([^"]+)"', block)
#             conf_m = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', block)
#             code = code_m.group(1) if code_m else ""
#             conf = float(conf_m.group(1)) if conf_m else 0.0
#             return code, conf, "regex_fallback"

#     def rerank(
#         self,
#         query: str,
#         shortlist: List[Dict[str, Any]],
#         blend_weight_llm: float = 0.6,
#     ) -> List[Dict[str, Any]]:
#         """Rerank a shortlist of products using LLaMA + embedding scores"""
#         if not shortlist:
#             return []

#         prompt = self._build_prompt(query, shortlist)
#         out = self.llm(
#             prompt,
#             max_tokens=self.max_tokens,
#             temperature=self.temperature,
#             top_p=self.top_p,
#             echo=False
#         )
#         text = out["choices"][0]["text"]
#         code, conf, reason = self._parse_json(text)

#         results = []
#         for p in shortlist:
#             product_code = str(p.get("code", "UNKNOWN")).lower()
#             embed_sim = float(p.get("score", 0.0))
#             llm_boost = conf if product_code == code.lower() else 0.0
#             final_score = blend_weight_llm * llm_boost + (1.0 - blend_weight_llm) * embed_sim

#             results.append({
#                 "code": p.get("code", "UNKNOWN"),
#                 "name": p.get("name", "UNKNOWN"),
#                 "desc": p.get("desc", "No description available"),
#                 "score": embed_sim,
#                 "llm_choice": product_code == code.lower(),
#                 "llm_confidence": conf if product_code == code.lower() else 0.0,
#                 "reason": reason if product_code == code.lower() else "",
#                 "final_score": final_score,
#             })

#         results.sort(key=lambda x: x["final_score"], reverse=True)
#         return results



import os
import json
import re
from typing import List, Dict, Any, Tuple

try:
    from llama_cpp import Llama
except ImportError:
    raise RuntimeError("llama-cpp-python is not installed. Install with: pip install llama-cpp-python")

class LlamaReranker:
    def __init__(
        self,
        model_path: str = os.path.join("models", "llama-2-7b-chat.Q4_K_M.gguf"),
        products_file: str = os.path.join("products", "products.json"),
        ctx_size: int = 2048,
        n_gpu_layers: int = 0,
        n_threads: int = None,
        temperature: float = 0.1,
        top_p: float = 0.9,
        max_tokens: int = 128,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"GGUF model not found at {model_path}.")
        if not os.path.exists(products_file):
            raise FileNotFoundError(f"Products file not found: {products_file}")

        # Load and flatten products
        with open(products_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.products: List[Dict[str, Any]] = []
        for ds_name, ds_info in raw.get("datasets", {}).items():
            self.products.append({
                "code": ds_name,
                "name": ds_name,
                "desc": ds_info.get("description", "")
            })
            for ind in ds_info.get("indicators", []):
                self.products.append({
                    "code": f"{ds_name}_{ind['name']}",
                    "name": ind['name'],
                    "desc": ind.get("description", "")
                })

        # Load LLaMA model
        self.llm = Llama(
            model_path=model_path,
            n_ctx=ctx_size,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads or os.cpu_count(),
            verbose=False,
        )

        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def _build_choice_block(self, candidates: List[Dict[str, Any]]) -> str:
        lines = []
        for i, p in enumerate(candidates, start=1):
            lines.append(f"{i}. CODE={p.get('code','UNKNOWN')}\n   NAME={p.get('name','UNKNOWN')}\n   DESC={p.get('desc','No description')}\n")
        return "\n".join(lines)

    def _build_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        choices = self._build_choice_block(candidates)
        system = "You are a product router. Pick the SINGLE best product CODE for the query. Output strict JSON."
        user = f"""Query: {query}

Catalog:
{choices}

Rules:
- Choose exactly ONE product code from the shortlist.
- Output strict JSON: {{"code": "<code>", "confidence": <0..1>, "reason": "<short reason>"}}"""

        return f"{system}\n\n{user}\n"

    def _parse_json(self, text: str) -> Tuple[str, float, str]:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return "", 0.0, "parse_failed"
        try:
            obj = json.loads(m.group(0))
            code = str(obj.get("code", "")).strip()
            conf = float(obj.get("confidence", 0.0))
            reason = str(obj.get("reason", "")).strip()
            conf = max(0.0, min(1.0, conf))
            return code, conf, reason
        except Exception:
            return "", 0.0, "json_error"

    def rerank(self, query: str, shortlist: List[Dict[str, Any]], blend_weight_llm: float = 0.6) -> List[Dict[str, Any]]:
        if not shortlist:
            return []

        prompt = self._build_prompt(query, shortlist)
        out = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p, echo=False)
        text = out["choices"][0]["text"]
        code, conf, reason = self._parse_json(text)

        results = []
        for p in shortlist:
            product_code = str(p.get("code", "UNKNOWN")).lower()
            embed_score = float(p.get("score", 0.0))
            llm_boost = conf if product_code == code.lower() else 0.0
            final_score = blend_weight_llm * llm_boost + (1 - blend_weight_llm) * embed_score

            results.append({
                "code": p.get("code", "UNKNOWN"),
                "name": p.get("name", "UNKNOWN"),
                "desc": p.get("desc", ""),
                "score": embed_score,
                "llm_choice": product_code == code.lower(),
                "llm_confidence": conf if product_code == code.lower() else 0.0,
                "reason": reason if product_code == code.lower() else "",
                "final_score": final_score,
            })

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
