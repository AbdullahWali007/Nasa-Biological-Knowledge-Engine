#!/usr/bin/env python3
"""
Engine Streamlit App — Improved search & accurate dataset summaries

Save as: engine_streamlit.py
Run: streamlit run engine_streamlit.py

Requirements:
    pip install pandas scikit-learn streamlit
    pip install google-genai        # optional for Gemini calls

Put index.json in same folder. Set GEMINI_API_KEY env var or edit GEMINI_API_KEY below.
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
import tempfile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import streamlit as st

# ---------------- CONFIG ----------------
INDEX_PATH = Path("index.json")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyBwdr7Uh2xpwuj2IQtoFY5EdzEOd_lPM1k")

TOP_K_DATASETS = 5
SIMILARITY_THRESHOLD = 0.02
ANALYSIS_ROWS = 1000        # how many rows to read for column stats (cap)
PREVIEW_ROWS = 5            # rows shown in quick preview
MAX_GEMINI_ATTEMPTS = 3
# ----------------------------------------

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("EngineStreamlit")

URL_REGEX = re.compile(r"(https?://[^\s'\"<>\)\]]+)", re.IGNORECASE)
NUMBER_REGEX = re.compile(r"\b(?:\d{2,}|[0-9]+(?:\.[0-9]+)?)\b")  # numbers with at least 2 digits or decimals


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
    row_count = len(df)
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
    Builds a TF-IDF index over datasets using:
      - dataset id
      - metadata fields
      - column names
      - textual sample rows
      - human-readable column summaries (top values & percentages)
    The index is built on startup and can be rebuilt.
    """
    def __init__(self, data_index: Dict[str, Any]):
        self.data_index = data_index or {}
        self.keys: List[str] = []
        self.corpus: List[str] = []
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
        # include column names & small textual preview + column summaries from a small analysis
        files = meta.get("files", {})
        file_count = 0
        for name, entry in list(files.items())[:4]:
            path = Path(entry if isinstance(entry, str) else (entry[0] if entry else ""))
            if not path.exists():
                continue
            # sample small number for search index (keep it faster)
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
                    # got df_preview
                    pass
            else:
                # we have df_preview
                # include column names & first few rows
                colnames = " ".join(map(str, df_preview.columns[:20]))
                parts.append(colnames)
                # include sample rows
                try:
                    sample_text = " ".join(df_preview.head(5).astype(str).agg(" ".join, axis=1).tolist())
                    parts.append(sample_text)
                    urls.extend(extract_urls(sample_text))
                except Exception:
                    pass
            file_count += 1
            if file_count >= 4:
                break

        # small fallback if nothing
        text = "\n".join(parts)
        return text, list(dict.fromkeys(urls))

    def _build_index(self):
        self.keys = []
        self.corpus = []
        urls_total = {}
        for ds, meta in self.data_index.items():
            txt, urls = self._make_dataset_text(ds, meta)
            self.keys.append(ds)
            self.corpus.append(txt if txt else ds)
            if urls:
                urls_total[ds] = urls
        # vectorize
        try:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
            self.matrix = self.vectorizer.fit_transform(self.corpus)
            logger.info("Search index built: %d datasets", len(self.keys))
        except Exception as e:
            logger.warning("Failed to build TF-IDF index: %s", e)
            self.vectorizer = None
            self.matrix = None

    def rebuild(self, data_index: Dict[str, Any]):
        self.data_index = data_index or {}
        self._build_index()

    def query(self, q: str, top_k: int = TOP_K_DATASETS) -> List[Dict[str, Any]]:
        """
        Return ranked datasets with combined scoring:
          - TF-IDF cosine sim (if available)
          - token overlap boost
          - number-match boost (if q contains numbers)
        """
        q_norm = q.strip()
        q_tokens = set(re.findall(r"\w+", q_norm.lower()))
        numbers_in_q = set(NUMBER_REGEX.findall(q_norm))
        scores = []
        # TF-IDF similarity
        if self.vectorizer and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([q_norm])
                sims = cosine_similarity(qv, self.matrix).flatten()
            except Exception:
                sims = np.zeros(len(self.keys))
        else:
            sims = np.zeros(len(self.keys))

        for idx, ds in enumerate(self.keys):
            sim = float(sims[idx]) if idx < len(sims) else 0.0
            # token overlap between query and corpus
            corpus_tokens = set(re.findall(r"\w+", (self.corpus[idx] or "").lower()))
            overlap = len(q_tokens & corpus_tokens) / (1 + len(q_tokens))
            score = 0.7 * sim + 0.3 * overlap
            # number match boost: look for numbers appearing in the corpus top values
            if numbers_in_q:
                found_num_boost = 0.0
                for num in numbers_in_q:
                    if num in (self.corpus[idx] or ""):
                        found_num_boost += 0.5
                score += found_num_boost
            scores.append((ds, float(score), float(sim), float(overlap)))
        # sort descending by score
        scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
        # filter by threshold but keep top_k
        results = []
        for ds, sc, simv, ov in scores_sorted[:max(top_k, len(scores_sorted))]:
            if sc >= SIMILARITY_THRESHOLD or len(results) < top_k:
                results.append({"dataset": ds, "score": sc, "sim": simv, "overlap": ov})
            if len(results) >= top_k:
                break
        return results


# ---------- Dataset context builder (improved) ----------
def build_context_for_dataset(ds: str, meta: Dict[str, Any], max_files: int = 4,
                              analysis_rows: int = ANALYSIS_ROWS, preview_rows: int = PREVIEW_ROWS) -> Tuple[str, List[str], Dict[str, Any]]:
    """
    Returns:
      - context string (human-readable with column summaries)
      - urls found in dataset
      - structured metadata including column summaries
    """
    urls = []
    blocks = []
    structured = {"dataset": ds, "columns": {}}

    # include metadata
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
        # attempt full analysis read (capped)
        df_analysis = read_table_for_analysis(path, nrows=analysis_rows)
        df_preview = safe_read_table(path, nrows=preview_rows)
        if df_preview is not None:
            blocks.append(f"FILE PREVIEW: {path.name}\n{df_preview.head(preview_rows).to_string(index=False)}")
        else:
            blocks.append(f"FILE: {path.name} (preview unreadable)")

        if df_analysis is None or df_analysis.empty:
            # try text read
            try:
                txt = path.read_text(encoding="utf-8", errors="ignore")[:2000]
                urls.extend(extract_urls(txt))
            except Exception:
                pass
            file_count += 1
            continue

        # analyze dataframe
        col_summ = analyze_dataframe(df_analysis, top_n=6)
        structured["columns"].update(col_summ)
        # produce human-readable column summaries for the context
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
    # dedupe urls
    deduped_urls = []
    for u in urls:
        if u not in deduped_urls:
            deduped_urls.append(u)
    return context_text, deduped_urls, structured


# ---------- Gemini integration (improved prompts) ----------
def call_gemini(prompt: str):
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        return False, f"Gemini SDK missing: {e}"

    if not GEMINI_API_KEY:
        return False, "Gemini API key missing."

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        return False, f"Failed to initialize genai.Client: {e}"

    attempt = 0
    while attempt < MAX_GEMINI_ATTEMPTS:
        attempt += 1
        try:
            logger.info("Gemini call attempt %d", attempt)
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                    max_output_tokens=1024,
                ),
            )
            return True, response
        except Exception as e:
            logger.warning("Gemini attempt %d failed: %s", attempt, e)
            if attempt >= MAX_GEMINI_ATTEMPTS:
                return False, f"Gemini failed after {attempt} attempts: {e}"
            time.sleep(1.5 * attempt)
    return False, "Gemini attempts exhausted"


def parse_gemini_response(resp: Any) -> Tuple[str, List[str], Dict[str, Any]]:
    try:
        candidates = getattr(resp, "candidates", None) or []
        if not candidates:
            text = getattr(resp, "text", None) or str(resp)
            return text, extract_urls(text), {"note": "no_candidates"}
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        texts, urls = [], []
        code_snippets = []
        code_outputs = []
        for p in parts:
            if getattr(p, "text", None):
                texts.append(p.text)
                urls += extract_urls(p.text)
            if getattr(p, "executable_code", None):
                code_snippets.append(getattr(p.executable_code, "code", ""))
            if getattr(p, "code_execution_result", None):
                code_outputs.append(getattr(p.code_execution_result, "output", ""))
        combined = "\n\n".join(texts).strip()
        # remove trailing JSON blocks for visible answer
        combined = re.sub(r"\n*\{[\s\S]*\}\s*$", "", combined).strip()
        meta = {"parts": len(parts), "code_snippets_count": len(code_snippets), "code_outputs_count": len(code_outputs)}
        return combined, list(dict.fromkeys(urls)), meta
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


# ---------- Improved fallback summarizer ----------
def local_fallback_summary(question: str, context_text: str, candidate_urls: List[str]) -> str:
    lines = [l.strip() for l in (context_text or "").splitlines() if l.strip()]
    important = [l for l in lines if any(k in l.lower() for k in ("flight", "space", "microgravity", "orbit", "condition", "control", "assay", "organism", "transcript", "methyl", "gene", "expression", "rna", "dna"))]
    if not important:
        important = lines[:6]
    summary = f"Local fallback summary for: {question}\n"
    if important:
        summary += "\n".join(f"- {l}" for l in important[:8])
    else:
        summary += "No local data available."
    if candidate_urls:
        summary += "\n\nLocal dataset URLs:\n" + "\n".join(candidate_urls[:6])
    
    # Add intelligent search suggestions
    summary += "\n\nSuggested NASA data sources to search:\n"
    nasa_keywords = ["NASA space biology", "microgravity research", "spaceflight transcriptomics", "NASA gene expression data"]
    for kw in nasa_keywords:
        if any(word in question.lower() for word in kw.split()):
            summary += f"- Search Google for: '{kw} dataset NASA Open Science'\n"
    
    return summary


# ---------- High-level ask flow combining everything ----------
class EngineRunner:
    def __init__(self, data_index: Dict[str, Any]):
        self.data_index = data_index or {}
        self.search_index = SearchIndex(self.data_index)

    def rebuild_search_index(self):
        self.search_index.rebuild(self.data_index)

    def ask(self, question: str, prefer_dataset_urls_first: bool = True) -> Dict[str, Any]:
        # find relevant datasets using improved index
        cand = self.search_index.query(question, top_k=TOP_K_DATASETS)
        used_datasets = cand
        contexts = []
        dataset_urls = []
        structured = {}
        for c in cand:
            ds = c["dataset"]
            meta = self.data_index.get(ds, {}) if isinstance(self.data_index, dict) else {}
            ctx, urls, struct = build_context_for_dataset(ds, meta)
            contexts.append(f"--- Dataset: {ds} (score={c['score']:.3f}) ---\n{ctx}")
            dataset_urls += urls
            structured[ds] = struct

        combined_context = "\n\n".join(contexts).strip()
        dataset_urls = list(dict.fromkeys(dataset_urls))

        # If no datasets or context empty, allow web search by telling Gemini
        use_web_search = not bool(combined_context)

        # Improved Gemini prompt with NASA-specific search instructions
        prompt = (
            "SYSTEM: You are a NASA Space Biology expert assistant with web search capabilities. "
            "Follow these guidelines precisely:\n\n"
            "1. FIRST analyze the provided DATA CONTEXT thoroughly\n"
            "2. If DATA CONTEXT contains relevant information, use it to answer the question\n"
            "3. If DATA CONTEXT is insufficient or missing, perform targeted web searches for:\n"
            "   - NASA Open Science Data Repository (data.nasa.gov)\n"
            "   - NASA GeneLab (genelab.nasa.gov)\n" 
            "   - NASA Space Biology Program data\n"
            "   - PubMed Central space biology studies\n"
            "   - Recent microgravity research publications\n\n"
            "4. Answer structure:\n"
            "   - Start with direct answer (2-3 sentences max)\n"
            "   - Mention which datasets were used (if any)\n"
            "   - List 3-5 most relevant URLs with brief descriptions\n"
            "   - Include specific NASA data sources when applicable\n\n"
            "5. CREDIBILITY: Prioritize .gov, .edu, NASA, and peer-reviewed sources\n"
            "6. FORMAT: Keep answer concise (<200 words), no JSON in visible answer\n\n"
            f"DATA CONTEXT:\n{combined_context or '[NO RELEVANT LOCAL DATASETS FOUND - PLEASE SEARCH NASA REPOSITORIES]'}\n\n"
            f"USER QUESTION:\n{question}\n\n"
            "Provide a focused answer with NASA data recommendations."
        )
        if use_web_search:
            prompt += "\n\nSEARCH REQUIRED: No relevant local data found. Please search NASA repositories and space biology databases."

        ok, resp = call_gemini(prompt)
        if not ok:
            logger.warning("Gemini failed: %s", resp)
            # fallback to improved local summary
            fallback = local_fallback_summary(question, combined_context, dataset_urls)
            return {
                "answer": fallback,
                "recommended_urls": dataset_urls,
                "used_datasets": used_datasets,
                "meta": {"fallback": True, "error": str(resp)},
                "structured": structured
            }

        text, urls_from_resp, meta = parse_gemini_response(resp)
        text = clean_answer_text(text)
        # prefer dataset urls first
        combined_urls = []
        if prefer_dataset_urls_first:
            for u in dataset_urls:
                if u not in combined_urls:
                    combined_urls.append(u)
            for u in urls_from_resp:
                if u not in combined_urls:
                    combined_urls.append(u)
        else:
            for u in urls_from_resp:
                if u not in combined_urls:
                    combined_urls.append(u)
            for u in dataset_urls:
                if u not in combined_urls:
                    combined_urls.append(u)
        return {
            "answer": text,
            "recommended_urls": combined_urls,
            "used_datasets": used_datasets,
            "meta": meta,
            "structured": structured
        }


# ---------- Helpers to load index ----------
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


# ---------- Streamlit App ----------
def main():
    st.set_page_config(
        page_title="Nasa Biological Space Engine — Intelligent NASA Space Biology Search",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'runner' not in st.session_state:
        data_index = load_index()
        st.session_state.runner = EngineRunner(data_index)
        st.session_state.logs = ["App initialized. Intelligent NASA Space Biology search engine loaded."]
        st.session_state.current_result = None
        st.session_state.selected_dataset = None
        
    
    # Header
    st.title("Nasa Biological Space Engine 🚀")
    st.subheader("Intelligent NASA Space Biology Search (A hack for common people and researchers)")
    
    # Sidebar
    with st.sidebar:
        st.header("Controls & Information")
        
        # Rebuild index button
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
        
        # Matched datasets
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
        
        # Logs
        st.subheader("📝 Logs")
        log_container = st.container(height=300)
        with log_container:
            for log_entry in st.session_state.logs[-10:]:  # Show last 10 logs
                st.text(log_entry)
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "🔍 Ask NASA Space Biology Question:",
            placeholder="e.g., 'What are the effects of microgravity on gene expression in mice?'",
            key="query_input"
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        ask_btn = st.button("🚀 Ask", use_container_width=True)
    
    # Process query
    if ask_btn and query:
        with st.spinner("🔬 Analyzing question for NASA space biology relevance..."):
            # Clear previous results
            st.session_state.current_result = None
            st.session_state.selected_dataset = None
            
            # Add to logs
            st.session_state.logs.append(f"Query: {query}")
            
            # Find relevant datasets
            with st.status("Searching NASA space biology datasets...", expanded=True) as status:
                cand = st.session_state.runner.search_index.query(query, top_k=TOP_K_DATASETS)
                for c in cand:
                    st.write(f"• {c['dataset']} (score: {c['score']:.3f})")
                status.update(label="Found relevant datasets", state="complete")
            
            # Get answer
            with st.status("Consulting NASA repositories and analyzing data...", expanded=True) as status:
                result = st.session_state.runner.ask(query, prefer_dataset_urls_first=True)
                st.session_state.current_result = result
                st.session_state.logs.append("NASA space biology analysis complete.")
                status.update(label="Analysis complete", state="complete")
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        # Create tabs for different result views
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Answer", "🔗 Recommended URLs", "📊 Dataset Details", "📋 Raw Metadata"])
        
        with tab1:
            st.subheader("Answer")
            st.write(result.get("answer", ""))
            
            # Copy answer button
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
            st.subheader("Dataset Details")
            
            # Show selected dataset preview
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
        
        with tab4:
            st.subheader("Raw Metadata")
            meta = result.get("meta", {})
            used_datasets = result.get("used_datasets", [])
            structured = result.get("structured", {})
            
            out_meta = {
                "meta": meta,
                "used_datasets": [d for d in used_datasets],
                "structured": structured
            }
            
            st.json(out_meta)
            
            # Download result button
            json_str = json.dumps(out_meta, indent=2, ensure_ascii=False)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="nasa_space_biology_result.json">💾 Download Result as JSON</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Dataset preview in main area if selected
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
