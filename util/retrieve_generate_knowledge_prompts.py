from __future__ import annotations

import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from util.fashion_knowledge_schema import (
    ALL_DIMS, STATIC_DIMS, TEMPORAL_DIMS, DEFAULT_PROMPTS_PER_DIM,
    dimension_output_count, fallback_attributes, get_query_variants,
    render_prompt, slugify,
)
from util.io_utils import ensure_dir, save_json, parse_bool


def _safe_json_extract(text: str) -> dict:
    text = (text or "").strip()
    # Direct parse first.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    # Extract first JSON object.
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _dedup_strings(xs: Iterable[str], max_n: int) -> list[str]:
    seen = set()
    out = []
    for x in xs:
        y = re.sub(r"\s+", " ", str(x).strip().strip("\"'`-•.;"))
        if not y or len(y) < 3:
            continue
        key = y.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(y)
        if len(out) >= max_n:
            break
    return out


class LocalTfidfRetriever:
    def __init__(self, passages: pd.DataFrame, max_features: int = 200_000,
                 ngram_max: int = 2, min_df: int = 2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.passages = passages.reset_index(drop=True)
        self.texts = self.passages["text"].fillna("").astype(str).tolist()
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=max_features,
            ngram_range=(1, ngram_max),
            min_df=min_df,
            dtype=np.float32,
        )
        print(f"[Retriever] Fitting TF-IDF on {len(self.texts)} Amazon evidence passages...", flush=True)
        self.X = self.vectorizer.fit_transform(self.texts)

    def search(self, query: str, topn: int = 20) -> pd.DataFrame:
        q = self.vectorizer.transform([query])
        scores = (self.X @ q.T).toarray().reshape(-1)
        if topn >= len(scores):
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, topn)[:topn]
            idx = idx[np.argsort(-scores[idx])]
        out = self.passages.iloc[idx].copy()
        out["score"] = scores[idx]
        out["query"] = query
        return out[out["score"] > 0].reset_index(drop=True)


def select_diverse_evidence(evidence: pd.DataFrame, max_evidence: int = 60) -> pd.DataFrame:
    if evidence.empty:
        return evidence
    evidence = evidence.sort_values("score", ascending=False).copy()
    evidence["_asin_rank"] = evidence.groupby("parent_asin").cumcount()
    # First pass: at most two passages per ASIN.
    selected = evidence[evidence["_asin_rank"] < 2].head(max_evidence)
    if len(selected) < max_evidence:
        extra = evidence[~evidence["passage_id"].isin(selected["passage_id"])].head(max_evidence - len(selected))
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected.drop(columns=["_asin_rank"], errors="ignore").drop_duplicates("passage_id").head(max_evidence)


def load_subjects(data_root: Path, prompt_subject: str, max_prompt_subjects: int | None,
                  min_subject_items: int = 5) -> list[tuple[str, str, int]]:
    hm = pd.read_parquet(data_root / "hm" / "processed" / "hm_item_text.parquet")
    if prompt_subject not in hm.columns:
        raise ValueError(f"prompt_subject={prompt_subject} not found in hm_item_text.parquet. Available: {list(hm.columns)}")
    s = hm[prompt_subject].fillna("Unknown").astype(str).str.strip()
    vc = s.value_counts()
    vc = vc[vc >= min_subject_items]
    if max_prompt_subjects is not None and max_prompt_subjects > 0:
        vc = vc.head(max_prompt_subjects)
    return [(str(k), slugify(k), int(v)) for k, v in vc.items()]


class QwenGenerator:
    def __init__(self, model_name: str, cuda_id: int = 0, mock: bool = False,
                 max_input_tokens: int = 4096, max_new_tokens: int = 256,
                 temperature: float = 0.2):
        self.mock = bool(mock)
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tokenizer = None
        self.model = None
        self.device = None
        if not self.mock:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map={"": self.device} if self.device.startswith("cuda") else None,
                trust_remote_code=True,
            )
            if self.device == "cpu":
                self.model.to(self.device)
            self.model.eval()

    def _chat(self, system: str, user: str) -> str:
        if self.mock:
            return "{}"
        import torch
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = system + "\n\n" + user + "\n"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                repetition_penalty=1.08,
                no_repeat_ngram_size=4,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def plan_queries(self, subject: str, dim: str, base_queries: list[str], n_queries: int) -> tuple[list[str], str]:
        if self.mock:
            return base_queries[:n_queries], "mock"
        system = "You are a local fashion knowledge retrieval agent. Return valid JSON only."
        user = f"""
We need retrieve Amazon fashion review evidence for the H&M category: {subject}
Knowledge dimension: {dim}
Seed queries:
{json.dumps(base_queries, ensure_ascii=False, indent=2)}

Generate {n_queries} diverse search queries. They should include concrete fashion terms and review keywords.
Return JSON only: {{"queries": ["..."]}}
""".strip()
        raw = self._chat(system, user)
        obj = _safe_json_extract(raw)
        qs = obj.get("queries", []) if isinstance(obj, dict) else []
        qs = _dedup_strings(qs, n_queries)
        if len(qs) < n_queries:
            qs.extend([q for q in base_queries if q not in qs])
        return qs[:n_queries], raw

    def generate_attributes(self, subject: str, dim: str, evidence: pd.DataFrame,
                            n_attrs: int) -> tuple[list[str], str, str]:
        if self.mock:
            return fallback_attributes(dim, n_attrs), "mock", "mock"
        snippets = []
        for i, (_, row) in enumerate(evidence.head(80).iterrows(), start=1):
            txt = str(row.get("text", ""))[:700]
            rating = row.get("rating", "")
            snippets.append(f"[{i}] rating={rating}; {txt}")
        system = "You generate compact fashion knowledge attribute phrases from retrieved Amazon evidence. Return valid JSON only."
        user = f"""
H&M category subject: {subject}
Knowledge dimension: {dim}

Retrieved Amazon evidence snippets:
{chr(10).join(snippets)}

Task:
Generate {n_attrs} diverse attribute phrases for this category and dimension.
Do NOT output full sentences. Do NOT include the category name. Do NOT invent unsupported facts.
Use short noun phrases that can fill a template like category + relation + attribute.
Return JSON only in this exact format:
{{"attributes": ["attribute phrase 1", "attribute phrase 2"]}}
""".strip()
        raw = self._chat(system, user)
        obj = _safe_json_extract(raw)
        attrs = obj.get("attributes", []) if isinstance(obj, dict) else []
        attrs = _dedup_strings(attrs, n_attrs)
        status = "ok" if len(attrs) >= max(1, min(3, n_attrs)) else "fallback"
        if len(attrs) < n_attrs:
            attrs.extend([x for x in fallback_attributes(dim, n_attrs) if x not in attrs])
        return attrs[:n_attrs], raw, status


def build_prompts(data_root: Path, prompt_subject: str = "product_group_name",
                  retrieve_mode: str = "local", retriever_type: str = "tfidf",
                  qwen_model: str = "Qwen/Qwen3-4B-Instruct-2507", cuda_id: int = 0,
                  max_prompt_subjects: int | None = None, min_subject_items: int = 5,
                  query_variants_per_dim: int = 6, retrieval_topn_per_query: int = 20,
                  evidence_max_per_dimension: int = 60, prompts_per_dim: str | None = None,
                  mock: bool = False, max_input_tokens: int = 4096, max_new_tokens: int = 256,
                  temperature: float = 0.2) -> None:
    cross_dir = data_root / "cross_domain"
    ensure_dir(cross_dir)
    pass_path = cross_dir / "amazon_evidence_passages.parquet"
    if not pass_path.exists():
        raise FileNotFoundError(f"Missing {pass_path}. Run build_amazon_evidence_corpus.sh first.")
    passages = pd.read_parquet(pass_path)
    passages["text"] = passages["text"].fillna("").astype(str)
    subjects = load_subjects(data_root, prompt_subject, max_prompt_subjects, min_subject_items)
    print(f"[Knowledge] subject={prompt_subject}, num_subjects={len(subjects)}, retrieve_mode={retrieve_mode}", flush=True)
    if retriever_type != "tfidf":
        print(f"[Knowledge] retriever_type={retriever_type} requested; current package defaults to local TF-IDF for stability.", flush=True)
    retriever = LocalTfidfRetriever(passages)
    gen = QwenGenerator(qwen_model, cuda_id=cuda_id, mock=mock, max_input_tokens=max_input_tokens,
                        max_new_tokens=max_new_tokens, temperature=temperature)

    cls_prompts = [[name, slug] for name, slug, _ in subjects]
    desc = {f"desc_{d}": [] for d in ALL_DIMS}
    long_rows = []
    evidence_rows = []
    raw_rows = []

    for subject, subject_slug, subject_count in tqdm(subjects, desc="Retrieve-generate fashion prompts"):
        per_subject = {d: [] for d in ALL_DIMS}
        for dim in ALL_DIMS:
            n_attrs = dimension_output_count(dim, prompts_per_dim)
            base_queries = get_query_variants(subject, dim, max_variants=query_variants_per_dim)
            query_raw = ""
            if retrieve_mode == "owl_agent":
                # Local Qwen3 query planner, modeled after STEAM's agentic query->retrieve->generate logic.
                queries, query_raw = gen.plan_queries(subject, dim, base_queries, query_variants_per_dim)
            else:
                queries = base_queries
            retrieved = []
            for q in queries:
                r = retriever.search(q, retrieval_topn_per_query)
                if len(r):
                    retrieved.append(r)
            if retrieved:
                ev = pd.concat(retrieved, ignore_index=True).drop_duplicates("passage_id")
                ev = select_diverse_evidence(ev, evidence_max_per_dimension)
            else:
                ev = pd.DataFrame(columns=list(passages.columns) + ["score", "query"])
            attrs, raw, status = gen.generate_attributes(subject, dim, ev, n_attrs)
            raw_rows.append({
                "subject": subject, "subject_slug": subject_slug, "dimension": dim,
                "retrieve_mode": retrieve_mode, "query_planner_raw": query_raw,
                "qwen_raw": raw, "parse_status": status,
                "num_evidence": int(len(ev)), "num_attributes": int(len(attrs)),
            })
            for rank, a in enumerate(attrs, start=1):
                prompt = render_prompt(subject, dim, a)
                per_subject[dim].append(prompt)
                split = "temporal" if dim in TEMPORAL_DIMS else "static"
                prompt_id = f"{subject_slug}__{dim}__{rank:02d}"
                long_rows.append({
                    "prompt_id": prompt_id,
                    "subject": subject,
                    "subject_slug": subject_slug,
                    "prompt_subject": prompt_subject,
                    "subject_item_count": subject_count,
                    "dimension": dim,
                    "split": split,
                    "attribute_phrase": a,
                    "knowledge_prompt": prompt,
                    "retrieve_mode": retrieve_mode,
                    "qwen_model": qwen_model,
                    "parse_status": status,
                })
                for _, er in ev.head(10).iterrows():
                    evidence_rows.append({
                        "prompt_id": prompt_id,
                        "subject": subject,
                        "dimension": dim,
                        "passage_id": er.get("passage_id"),
                        "parent_asin": er.get("parent_asin"),
                        "score": float(er.get("score", 0.0)),
                        "rating": er.get("rating", None),
                        "review_text": str(er.get("review_text", ""))[:500],
                    })
        for dim in ALL_DIMS:
            desc[f"desc_{dim}"].append(per_subject[dim])

    prompts = pd.DataFrame(long_rows)
    static = prompts[prompts["split"] == "static"].copy()
    temporal = prompts[prompts["split"] == "temporal"].copy()
    static.to_parquet(cross_dir / "knowledge_prompts_static.parquet", index=False)
    temporal.to_parquet(cross_dir / "knowledge_prompts_temporal.parquet", index=False)
    prompts.to_parquet(cross_dir / "knowledge_prompts_all.parquet", index=False)
    pd.DataFrame(evidence_rows).to_parquet(cross_dir / "knowledge_prompt_evidence.parquet", index=False)
    pd.DataFrame(raw_rows).to_parquet(cross_dir / "knowledge_prompt_qwen_raw.parquet", index=False)

    med_style = {"metadata": {
        "prompt_subject": prompt_subject,
        "retrieve_mode": retrieve_mode,
        "retriever_type": retriever_type,
        "qwen_model": qwen_model,
        "temporal_prompts_enabled_for_default_training": False,
        "note": "Static prompts are for default 7-day prediction training. Temporal prompts are saved for social analysis or social-task fine-tuning only.",
    }, "cls_prompts": cls_prompts, **desc}
    save_json(med_style, cross_dir / "fashion_knowledge_prompts.json")
    # JSONL preview for easier grep.
    with (cross_dir / "fashion_knowledge_prompts_long.jsonl").open("w", encoding="utf-8") as f:
        for rec in long_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(static)} static prompts and {len(temporal)} temporal prompts to {cross_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="./data")
    p.add_argument("--prompt_subject", choices=["product_group_name", "product_type_name"], default="product_group_name")
    p.add_argument("--retrieve_mode", choices=["local", "owl_agent"], default="local")
    p.add_argument("--retriever_type", default="tfidf")
    p.add_argument("--qwen_model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--cuda_id", type=int, default=0)
    p.add_argument("--max_prompt_subjects", type=int, default=None)
    p.add_argument("--min_subject_items", type=int, default=5)
    p.add_argument("--query_variants_per_dim", type=int, default=6)
    p.add_argument("--retrieval_topn_per_query", type=int, default=20)
    p.add_argument("--evidence_max_per_dimension", type=int, default=60)
    p.add_argument("--prompts_per_dim", default=None, help="Override counts, e.g. material:10,design:12,value:8")
    p.add_argument("--mock", type=parse_bool, default=False)
    p.add_argument("--max_input_tokens", type=int, default=4096)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()
    d = vars(args).copy()
    d["data_root"] = Path(d["data_root"])
    build_prompts(**d)


if __name__ == "__main__":
    main()
