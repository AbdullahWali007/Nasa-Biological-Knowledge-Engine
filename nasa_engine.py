#!/usr/bin/env python3
"""
Engine Streamlit App — Improved search & accurate dataset summaries with PMC publications

Save as: engine_streamlit.py
Run: streamlit run engine_streamlit.py

Requirements:
    pip install pandas scikit-learn streamlit openai

Put index.json and SB_publication_PMC.json in same folder.
"""

from __future__ import annotations
import os
import re
import json
import time
import logging
import webbrowser
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import base64

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st
from openai import OpenAI  

# ---------------- CONFIG ----------------
INDEX_PATH = Path("index.json")
PUBLICATIONS_PATH = Path("SB_publication_PMC.json")

# LongCat Configuration
LONGCAT_MODEL = "LongCat-Flash-Chat" 
LONGCAT_BASE_URL = "https://api.longcat.chat/openai/v1"
# We check secrets first, but fallback to your provided key
DEFAULT_LONGCAT_KEY = 'ak_1ZZ4nn04h3GB7Hj2Zm2g36nx0mD2J' 

TOP_K_DATASETS = 5
TOP_K_PUBLICATIONS = 8
SIMILARITY_THRESHOLD = 0.02
ANALYSIS_ROWS = 1000        # how many rows to read for column stats (cap)
PREVIEW_ROWS = 5            # rows shown in quick preview
MAX_API_ATTEMPTS = 3
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("EngineStreamlit")

URL_REGEX = re.compile(r"(https?://[^\s'\"<>\)\]]+)", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b(?:\d{2,}|[0-9]+(?:\.[0-9]+)?)\b")


# ---------- Utilities ----------
def extract_urls(text: str) -> List[str]:
    if not text:
        return []
    found = URL_REGEX.findall(text)
    seen = set()
    out = []
    for u in found:
        u = u.rstrip(".,;:)")
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def safe_read_table(path: Path, nrows: int = PREVIEW_ROWS) -> Optional[pd.DataFrame]:
    """Try reading a table; return None if fails. Use small memory-friendly reads."""
    for sep in ("\t", ","):
        try:
            return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
        except Exception:
            continue
    return None


def read_table_for_analysis(path: Path, nrows: int = ANALYSIS_ROWS) -> Optional[pd.DataFrame]:
    """Read more rows for statistical analysis. Falls back gracefully."""
    for sep in ("\t", ","):
        try:
            return pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
        except Exception:
            continue
    return None


def is_numeric_series(s: pd.Series) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(s)
    except Exception:
        # fallback: attempt to coerce
        try:
            pd.to_numeric(s.dropna().iloc[:50])
            return True
        except Exception:
            return False


def analyze_dataframe(df: pd.DataFrame, top_n: int = 6) -> Dict[str, Any]:
    """
    Compute column-level summaries:
      - For numeric columns: count, mean, median, std, min, max, unique_count, top frequencies
      - For non-numeric: unique_count, top frequencies
    Returns dict: {col_name: {summary...}}
    """
    summaries = {}
    for col in df.columns:
        try:
            series = df[col].dropna()
            if series.empty:
                summaries[col] = {"note": "empty", "count": 0}
                continue
            # try numeric detection
            numeric = is_numeric_series(series)
            info: Dict[str, Any] = {"count": int(len(series)), "unique": int(series.nunique())}
            if numeric:
                # coerce numeric
                snum = pd.to_numeric(series, errors="coerce").dropna()
                if snum.empty:
                    numeric = False
                else:
                    info.update({
                        "numeric": True,
                        "mean": float(snum.mean()),
                        "median": float(snum.median()),
                        "std": float(snum.std()) if len(snum) > 1 else 0.0,
                        "min": float(snum.min()),
                        "max": float(snum.max()),
                    })
                    # compute top frequencies
                    vc = snum.value_counts().head(top_n)
                    top = []
                    total = len(snum)
                    for val, cnt in vc.items():
                        try:
                            value = float(str(val))
                        except (ValueError, TypeError):
                            value = val
                        top.append({"value": value, "count": int(cnt), "pct": round(100.0 * cnt / total, 2)})
                    info["top"] = top
            if not numeric:
                # treat as categorical/text
                vc = series.astype(str).value_counts().head(top_n)
                total = len(series)
                top = []
                for val, cnt in vc.items():
                    top.append({"value": str(val), "count": int(cnt), "pct": round(100.0 * cnt / total, 2)})
                info["top"] = top
            summaries[col] = info
        except Exception as e:
            summaries[col] = {"error": str(e)}
    return summaries


# ---------- Search index building ----------
class SearchIndex:
    """
    Builds a TF-IDF index over datasets and publications.
    """
    def __init__(self, data_index: Dict[str, Any], publications: List[Dict[str, str]]):
        self.data_index = data_index or {}
        self.publications = publications or []
        self.keys: List[str] = []
        self.corpus: List[str] = []
        self.entry_types: List[str] = []  # "dataset" or "publication"
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self._build_index()

    @staticmethod
    def _make_dataset_text(ds: str, meta: Dict[str, Any]) -> Tuple[str, List[str]]:
        parts = [f"DATASET_ID: {ds}"]
        urls: List[str] = []
        # include metadata fields
        for k in ("organism", "assay_types", "description", "folder"):
            v = meta.get(k)
            if v:
                if isinstance(v, list):
                    parts.append(" ".join(map(str, v)))
                else:
                    parts.append(str(v))
                if isinstance(v, str):
                    urls.extend(extract_urls(v))
        # include column names & small textual preview
        files = meta.get("files", {})
        file_count = 0
        for name, entry in list(files.items())[:4]:
            path = Path(entry if isinstance(entry, str) else (entry[0] if entry else ""))
            if not path.exists():
                continue
            try:
                df_preview = pd.read_csv(path, sep=None, nrows=50, engine="python", low_memory=False)
            except Exception:
                try:
                    df_preview = pd.read_csv(path, nrows=20, low_memory=False)
                except Exception:
                    # try read as text
                    try:
                        txt = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                        parts.append(txt)
                        urls.extend(extract_urls(txt))
                    except Exception:
                        pass
                else:
                    pass
            else:
                # include column names & first few rows
                colnames = " ".join(map(str, df_preview.columns[:20]))
                parts.append(colnames)
                try:
                    sample_text = " ".join(df_preview.head(5).astype(str).agg(" ".join, axis=1).tolist())
                    parts.append(sample_text)
                    urls.extend(extract_urls(sample_text))
                except Exception:
                    pass
            file_count += 1
            if file_count >= 4:
                break

        text = "\n".join(parts)
        return text, list(dict.fromkeys(urls))

    def _build_index(self):
        self.keys = []
        self.corpus = []
        self.entry_types = []
        
        # Index datasets
        for ds, meta in self.data_index.items():
            txt, urls = self._make_dataset_text(ds, meta)
            self.keys.append(ds)
            self.corpus.append(txt if txt else ds)
            self.entry_types.append("dataset")

        # Index publications
        for idx, pub in enumerate(self.publications):
            title = pub.get("Title", "")
            link = pub.get("Link", "")
            pub_text = f"PUBLICATION: {title} {link}"
            self.keys.append(f"pub_{idx}")
            self.corpus.append(pub_text)
            self.entry_types.append("publication")

        # vectorize
        try:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000)
            self.matrix = self.vectorizer.fit_transform(self.corpus)
            logger.info("Search index built: %d datasets, %d publications", 
                       len([t for t in self.entry_types if t == "dataset"]),
                       len([t for t in self.entry_types if t == "publication"]))
        except Exception as e:
            logger.warning("Failed to build TF-IDF index: %s", e)
            self.vectorizer = None
            self.matrix = None

    def rebuild(self, data_index: Dict[str, Any], publications: List[Dict[str, str]]):
        self.data_index = data_index or {}
        self.publications = publications or []
        self._build_index()

    def query(self, q: str, top_k_datasets: int = TOP_K_DATASETS, top_k_publications: int = TOP_K_PUBLICATIONS) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        q_norm = q.strip()
        q_tokens = set(re.findall(r"\w+", q_norm.lower()))
        numbers_in_q = set(NUMBER_REGEX.findall(q_norm))
        scores = []
        
        if self.vectorizer and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([q_norm])
                sims = cosine_similarity(qv, self.matrix).flatten()
            except Exception:
                sims = np.zeros(len(self.keys))
        else:
            sims = np.zeros(len(self.keys))

        for idx, key in enumerate(self.keys):
            entry_type = self.entry_types[idx]
            sim = float(sims[idx]) if idx < len(sims) else 0.0
            corpus_tokens = set(re.findall(r"\w+", (self.corpus[idx] or "").lower()))
            overlap = len(q_tokens & corpus_tokens) / (1 + len(q_tokens))
            score = 0.7 * sim + 0.3 * overlap
            if numbers_in_q:
                found_num_boost = 0.0
                for num in numbers_in_q:
                    if num in (self.corpus[idx] or ""):
                        found_num_boost += 0.5
                score += found_num_boost
            scores.append((key, entry_type, float(score), float(sim), float(overlap)))
        
        scores_sorted = sorted(scores, key=lambda x: x[2], reverse=True)
        
        dataset_results = []
        publication_results = []
        
        for key, entry_type, sc, simv, ov in scores_sorted:
            if entry_type == "dataset" and sc >= SIMILARITY_THRESHOLD and len(dataset_results) < top_k_datasets:
                dataset_results.append({"dataset": key, "score": sc, "sim": simv, "overlap": ov})
            elif entry_type == "publication" and sc >= SIMILARITY_THRESHOLD and len(publication_results) < top_k_publications:
                pub_idx = int(key.replace("pub_", ""))
                if pub_idx < len(self.publications):
                    pub = self.publications[pub_idx]
                    publication_results.append({
                        "title": pub.get("Title", ""),
                        "link": pub.get("Link", ""),
                        "score": sc,
                        "sim": simv,
                        "overlap": ov
                    })

        return dataset_results, publication_results


# ---------- Dataset context builder ----------
def build_context_for_dataset(ds: str, meta: Dict[str, Any], max_files: int = 4,
                              analysis_rows: int = ANALYSIS_ROWS, preview_rows: int = PREVIEW_ROWS) -> Tuple[str, List[str], Dict[str, Any]]:
    urls = []
    blocks = []
    structured = {"dataset": ds, "columns": {}}

    desc = meta.get("description", "")
    structured["description"] = desc
    if desc:
        urls.extend(extract_urls(str(desc)))
        blocks.append(f"Description: {desc}")

    files = meta.get("files", {}) if isinstance(meta, dict) else {}
    file_count = 0
    for name, entry in list(files.items())[:max_files]:
        path = Path(entry if isinstance(entry, str) else (entry[0] if entry else ""))
        if not path.exists():
            blocks.append(f"FILE: {name} - (missing)")
            file_count += 1
            continue
        
        df_analysis = read_table_for_analysis(path, nrows=analysis_rows)
        df_preview = safe_read_table(path, nrows=preview_rows)
        if df_preview is not None:
            blocks.append(f"FILE PREVIEW: {path.name}\n{df_preview.head(preview_rows).to_string(index=False)}")
        else:
            blocks.append(f"FILE: {path.name} (preview unreadable)")

        if df_analysis is None or df_analysis.empty:
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                urls.extend(extract_urls(txt))
            except Exception:
                pass
            file_count += 1
            continue

        col_summ = analyze_dataframe(df_analysis, top_n=6)
        structured["columns"].update(col_summ)
        for col, info in col_summ.items():
            if info.get("count", 0) == 0:
                continue
            if info.get("numeric"):
                top_vals = info.get("top", [])
                top_str = ", ".join(f"{tv['value']} ({tv['pct']}%)" for tv in top_vals[:4]) if top_vals else ""
                blocks.append(f"Column: {col} [numeric] count={info.get('count')} mean={info.get('mean'):.3f} median={info.get('median'):.3f} std={info.get('std'):.3f} top={top_str}")
                for tv in top_vals:
                    urls.extend(extract_urls(str(tv.get("value"))))
            else:
                top_vals = info.get("top", [])
                top_str = ", ".join(f"'{tv['value']}' ({tv['pct']}%)" for tv in top_vals[:6]) if top_vals else ""
                blocks.append(f"Column: {col} [categorical] count={info.get('count')} unique={info.get('unique')} top={top_str}")
                for tv in top_vals:
                    urls.extend(extract_urls(str(tv.get("value"))))
        file_count += 1

    context_text = "\n".join(blocks)
    deduped_urls = []
    for u in urls:
        if u not in deduped_urls:
            deduped_urls.append(u)
    return context_text, deduped_urls, structured


# ---------- LongCat API Integration ----------
def call_longcat(prompt: str) -> Tuple[bool, Any]:
    """
    Connects to LongCat AI via the OpenAI SDK compatibility layer.
    """
    # 1. Get API Key
    api_key = st.secrets.get("LONGCAT_API_KEY", DEFAULT_LONGCAT_KEY)
    
    if not api_key:
        return False, "LongCat API key missing. Please check secrets or configuration."

    # 2. Initialize Client
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=LONGCAT_BASE_URL
        )
    except Exception as e:
        return False, f"Failed to initialize OpenAI/LongCat Client: {e}"

    # 3. Call API
    attempt = 0
    while attempt < MAX_API_ATTEMPTS:
        attempt += 1
        try:
            logger.info("LongCat call attempt %d", attempt)
            
            response = client.chat.completions.create(
                model=LONGCAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful NASA Space Biology expert assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )
            return True, response
        except Exception as e:
            logger.warning("LongCat attempt %d failed: %s", attempt, e)
            if attempt >= MAX_API_ATTEMPTS:
                return False, f"LongCat failed after {attempt} attempts: {e}"
            time.sleep(1.5 * attempt)
            
    return False, "API attempts exhausted"


def parse_longcat_response(resp: Any) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Parses the OpenAI-style response object from LongCat.
    """
    try:
        # Extract content from the ChatCompletion object
        if hasattr(resp, 'choices') and len(resp.choices) > 0:
            content = resp.choices[0].message.content
        else:
            # Fallback for weird responses or errors passed as strings
            content = str(resp)

        if not content:
            content = "No content returned from API."

        # Extract URLs from the answer text
        urls = extract_urls(content)
        
        # Meta info
        meta = {
            "model": getattr(resp, 'model', 'unknown'),
            "usage": dict(resp.usage) if hasattr(resp, 'usage') else {}
        }
        
        return content, list(dict.fromkeys(urls)), meta

    except Exception as e:
        logger.exception("Parse error")
        return str(resp), extract_urls(str(resp)), {"parse_error": str(e)}


def clean_answer_text(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"[\x00-\x1f]", " ", text)
    t = re.sub(r"As an AI language model[^\n.]*[.\n]?", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


# ---------- Fallback summarizer ----------
def local_fallback_summary(question: str, context_text: str, candidate_urls: List[str], publications: List[Dict[str, Any]]) -> str:
    lines = [l.strip() for l in (context_text or "").splitlines() if l.strip()]
    important = [l for l in lines if any(k in l.lower() for k in ("flight", "space", "microgravity", "orbit", "condition", "control", "assay", "organism", "transcript", "methyl", "gene", "expression", "rna", "dna"))]
    if not important:
        important = lines[:6]
    summary = f"Local fallback summary for: {question}\n"
    if important:
        summary += "\n".join(f"- {l}" for l in important[:8])
    else:
        summary += "No local data available."
    
    if publications:
        summary += "\n\nRelated Research Publications:\n"
        for pub in publications[:5]:
            summary += f"- {pub.get('title', '')}\n  {pub.get('link', '')}\n"
    
    if candidate_urls:
        summary += "\n\nLocal dataset URLs:\n" + "\n".join(candidate_urls[:6])
    
    summary += "\n\nSuggested NASA data sources to search:\n"
    nasa_keywords = ["NASA space biology", "microgravity research", "spaceflight transcriptomics", "NASA gene expression data"]
    for kw in nasa_keywords:
        if any(word in question.lower() for word in kw.split()):
            summary += f"- Search Google for: '{kw} dataset NASA Open Science'\n"
    
    return summary


# ---------- High-level ask flow ----------
class EngineRunner:
    def __init__(self, data_index: Dict[str, Any], publications: List[Dict[str, str]]):
        self.data_index = data_index or {}
        self.publications = publications or []
        self.search_index = SearchIndex(self.data_index, self.publications)

    def rebuild_search_index(self):
        self.search_index.rebuild(self.data_index, self.publications)

    def ask(self, question: str, prefer_dataset_urls_first: bool = True) -> Dict[str, Any]:
        # find relevant datasets
        dataset_candidates, publication_candidates = self.search_index.query(
            question, 
            top_k_datasets=TOP_K_DATASETS, 
            top_k_publications=TOP_K_PUBLICATIONS
        )
        
        used_datasets = dataset_candidates
        used_publications = publication_candidates
        
        contexts = []
        dataset_urls = []
        structured = {}
        
        # Build context for datasets
        for c in dataset_candidates:
            ds = c["dataset"]
            meta = self.data_index.get(ds, {}) if isinstance(self.data_index, dict) else {}
            ctx, urls, struct = build_context_for_dataset(ds, meta)
            contexts.append(f"--- Dataset: {ds} (score={c['score']:.3f}) ---\n{ctx}")
            dataset_urls += urls
            structured[ds] = struct

        # Add publications to context
        if publication_candidates:
            contexts.append("--- Related Research Publications ---")
            for pub in publication_candidates:
                contexts.append(f"Publication: {pub['title']}\nLink: {pub['link']}\nRelevance Score: {pub['score']:.3f}")

        combined_context = "\n\n".join(contexts).strip()
        dataset_urls = list(dict.fromkeys(dataset_urls))

        use_web_search = not bool(combined_context)

        # Prompt for LongCat
        prompt = (
            "SYSTEM: You are a NASA Space Biology expert assistant. "
            "Follow these guidelines precisely:\n\n"
            "1. FIRST analyze the provided DATA CONTEXT including datasets and research publications\n"
            "2. If DATA CONTEXT contains relevant information, use it to answer the question\n"
            "3. If DATA CONTEXT is insufficient, suggest targeted searches for:\n"
            "   - NASA Open Science Data Repository (data.nasa.gov)\n"
            "   - NASA GeneLab (genelab.nasa.gov)\n" 
            "   - PubMed Central\n\n"
            "4. Answer structure:\n"
            "   - Start with direct answer (2-3 sentences max)\n"
            "   - Mention which datasets and publications were used (if any)\n"
            "   - List 3-5 most relevant URLs with brief descriptions\n"
            "   - Include specific NASA data sources when applicable\n\n"
            "5. CREDIBILITY: Prioritize .gov, .edu, NASA, and peer-reviewed sources\n"
            "6. FORMAT: Keep answer concise (<200 words), no JSON in visible answer\n\n"
            f"DATA CONTEXT:\n{combined_context or '[NO RELEVANT LOCAL DATASETS OR PUBLICATIONS FOUND]'}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Provide a focused answer with NASA data recommendations and relevant publications."
        )
        if use_web_search:
            prompt += "\n\nNote: No local data found. Please rely on your general knowledge and suggest external NASA resources."

        ok, resp = call_longcat(prompt)
        if not ok:
            logger.warning("LongCat API failed: %s", resp)
            fallback = local_fallback_summary(question, combined_context, dataset_urls, used_publications)
            return {
                "answer": fallback,
                "recommended_urls": dataset_urls,
                "used_datasets": used_datasets,
                "used_publications": used_publications,
                "meta": {"fallback": True, "error": str(resp)},
                "structured": structured
            }

        text, urls_from_resp, meta = parse_longcat_response(resp)
        text = clean_answer_text(text)
        
        # Combine URLs
        combined_urls = []
        if prefer_dataset_urls_first:
            for u in dataset_urls:
                if u not in combined_urls:
                    combined_urls.append(u)
            for pub in used_publications:
                if pub.get("link") and pub["link"] not in combined_urls:
                    combined_urls.append(pub["link"])
            for u in urls_from_resp:
                if u not in combined_urls:
                    combined_urls.append(u)
        else:
            for u in urls_from_resp:
                if u not in combined_urls:
                    combined_urls.append(u)
            for pub in used_publications:
                if pub.get("link") and pub["link"] not in combined_urls:
                    combined_urls.append(pub["link"])
            for u in dataset_urls:
                if u not in combined_urls:
                    combined_urls.append(u)
                    
        return {
            "answer": text,
            "recommended_urls": combined_urls,
            "used_datasets": used_datasets,
            "used_publications": used_publications,
            "meta": meta,
            "structured": structured
        }


# ---------- Helpers to load index and publications ----------
def load_index() -> Dict[str, Any]:
    if not INDEX_PATH.exists():
        logger.warning("index.json not found; App will start with empty index.")
        return {}
    try:
        with INDEX_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.exception("Failed to load index.json")
        return {}


def load_publications() -> List[Dict[str, str]]:
    if not PUBLICATIONS_PATH.exists():
        logger.warning("SB_publication_PMC.json not found; App will start without publications.")
        return []
    try:
        with PUBLICATIONS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.exception("Failed to load SB_publication_PMC.json")
        return []


# ---------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="Nasa Biological Space Engine — Intelligent NASA Space Biology Search",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    if 'runner' not in st.session_state:
        data_index = load_index()
        publications = load_publications()
        st.session_state.runner = EngineRunner(data_index, publications)
        st.session_state.logs = ["App initialized. Intelligent NASA Space Biology search engine loaded."]
        st.session_state.current_result = None
        st.session_state.selected_dataset = None
        
    
    st.title("Nasa Biological Space Engine 🚀")
    st.subheader("Intelligent NASA Space Biology Search (Powered by LongCat AI)")
    
    with st.sidebar:
        st.header("Controls & Information")
        if st.button("🔁 Rebuild Search Index", use_container_width=True):
            with st.spinner("Rebuilding NASA space biology search index..."):
                try:
                    st.session_state.runner.rebuild_search_index()
                    st.session_state.logs.append("Search index rebuilt.")
                    st.success("NASA space biology search index rebuilt.")
                except Exception as e:
                    st.error(f"Could not rebuild index: {e}")
                    st.session_state.logs.append(f"Rebuild failed: {e}")
        
        st.divider()
        st.subheader("📊 Matched Datasets")
        if st.session_state.current_result:
            datasets = st.session_state.current_result.get("used_datasets", [])
            dataset_options = [f"{d['dataset']} (score={d['score']:.3f})" for d in datasets]
            if dataset_options:
                selected = st.selectbox("Select dataset for preview:", dataset_options)
                if selected:
                    st.session_state.selected_dataset = selected.split()[0]
            else:
                st.info("No datasets matched")
        else:
            st.info("Run a query to see matched datasets")
        
        st.divider()
        st.subheader("📝 Logs")
        log_container = st.container(height=300)
        with log_container:
            for log_entry in st.session_state.logs[-10:]:
                st.text(log_entry)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "🔍 Ask NASA Space Biology Question:",
            placeholder="e.g., 'What are the effects of microgravity on gene expression in mice?'",
            key="query_input"
        )
    with col2:
        st.write("")
        st.write("")
        ask_btn = st.button("🚀 Ask", use_container_width=True)
    
    if ask_btn and query:
        with st.spinner("🔬 Analyzing question for NASA space biology relevance..."):
            st.session_state.current_result = None
            st.session_state.selected_dataset = None
            st.session_state.logs.append(f"Query: {query}")
            
            with st.status("Searching NASA space biology datasets and publications...", expanded=True) as status:
                dataset_cand, publication_cand = st.session_state.runner.search_index.query(
                    query, 
                    top_k_datasets=TOP_K_DATASETS, 
                    top_k_publications=TOP_K_PUBLICATIONS
                )
                st.write("**Datasets:**")
                for c in dataset_cand:
                    st.write(f"• {c['dataset']} (score: {c['score']:.3f})")
                st.write("**Publications:**")
                for c in publication_cand[:3]:
                    st.write(f"• {c['title'][:80]}... (score: {c['score']:.3f})")
                status.update(label="Found relevant datasets and publications", state="complete")
            
            with st.status("Consulting LongCat AI and analyzing data...", expanded=True) as status:
                result = st.session_state.runner.ask(query, prefer_dataset_urls_first=True)
                st.session_state.current_result = result
                st.session_state.logs.append("NASA space biology analysis complete.")
                status.update(label="Analysis complete", state="complete")
    
    if st.session_state.current_result:
        result = st.session_state.current_result
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Answer", "🔗 Recommended URLs", "📚 Research Publications", "📊 Dataset Details", "📋 Raw Metadata"])
        
        with tab1:
            st.subheader("Answer")
            st.write(result.get("answer", ""))
            if st.button("📋 Copy Answer", key="copy_answer"):
                st.code(result.get("answer", ""), language=None)
                st.success("Answer copied to clipboard!")
        
        with tab2:
            st.subheader("Recommended URLs")
            urls = result.get("recommended_urls", [])
            if urls:
                for i, url in enumerate(urls):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"{i+1}. {url}")
                    with col2:
                        if st.button("🌐 Open", key=f"open_{i}"):
                            webbrowser.open_new_tab(url)
            else:
                st.info("No URLs recommended")
        
        with tab3:
            st.subheader("Research Publications")
            publications = result.get("used_publications", [])
            if publications:
                for i, pub in enumerate(publications):
                    with st.expander(f"📄 {pub.get('title', 'Unknown Title')}", expanded=i < 2):
                        st.write(f"**Title:** {pub.get('title', 'N/A')}")
                        st.write(f"**Link:** {pub.get('link', 'N/A')}")
                        st.write(f"**Relevance Score:** {pub.get('score', 0):.3f}")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            if st.button("🌐 Open Publication", key=f"open_pub_{i}"):
                                webbrowser.open_new_tab(pub.get('link', ''))
                        with col2:
                            if st.button("📋 Copy Link", key=f"copy_pub_{i}"):
                                st.code(pub.get('link', ''), language=None)
                                st.success("Link copied to clipboard!")
            else:
                st.info("No relevant publications found")
        
        with tab4:
            st.subheader("Dataset Details")
            if st.session_state.selected_dataset:
                ds = st.session_state.selected_dataset
                meta = st.session_state.runner.data_index.get(ds, {})
                with st.expander(f"📁 Dataset: {ds}", expanded=True):
                    ctx, urls, struct = build_context_for_dataset(ds, meta)
                    st.subheader("Context Summary")
                    st.text_area("Context", ctx, height=200, key=f"ctx_{ds}")
                    if urls:
                        st.subheader("Dataset URLs")
                        for url in urls:
                            st.write(f"• {url}")
                    st.subheader("Column Summaries")
                    st.json(struct)
            else:
                st.info("Select a dataset from the sidebar to view details")
        
        with tab5:
            st.subheader("Raw Metadata")
            out_meta = {
                "meta": result.get("meta", {}),
                "used_datasets": result.get("used_datasets", []),
                "used_publications": result.get("used_publications", []),
                "structured": result.get("structured", {})
            }
            st.json(out_meta)
            json_str = json.dumps(out_meta, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="nasa_space_biology_result.json">💾 Download Result as JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    elif st.session_state.selected_dataset and not st.session_state.current_result:
        ds = st.session_state.selected_dataset
        meta = st.session_state.runner.data_index.get(ds, {})
        st.subheader(f"📁 Dataset Preview: {ds}")
        ctx, urls, struct = build_context_for_dataset(ds, meta)
        with st.expander("Context Summary", expanded=True):
            st.text_area("Context", ctx, height=200)
        if urls:
            with st.expander("Dataset URLs", expanded=True):
                for url in urls:
                    st.write(f"• {url}")
        with st.expander("Column Summaries", expanded=True):
            st.json(struct)


if __name__ == "__main__":
    main()