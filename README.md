# 🚀 NASA Biological Space Engine

**An intelligent NASA Space Biology search engine built with Streamlit, TF-IDF search.**
This tool helps researchers and curious minds explore biological datasets (microgravity, spaceflight, gene expression, etc.) with dataset previews, statistical summaries, and smart querying.

---

## ✨ Features

* 🔍 **Intelligent Search** using TF-IDF with overlap + number boosts.
* 📊 **Dataset Summaries** with column-level statistics (mean, median, std, frequencies).
* 🌐 **Advanced Search** Advanced answers + search recommendations.
* 🧪 **Summarizer** For Really Large Search Results, it automatically Summarises the Result.
* 📂 **Dataset Preview** with first few rows and column summaries.
* 🎛️ **Streamlit Interface** with sidebar controls, logs, and interactive dataset browsing.

---

## 📦 Requirements

Install dependencies:

```bash
pip install pandas scikit-learn streamlit
pip install google-genai   # optional, for Gemini API calls
```

---

## ▶️ Usage

1. Place your dataset `index.json` in the same directory.
2. (Optional) Set your **Gemini API key**:

   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```
3. Run the app:

   ```bash
   streamlit run engine_deepseek.py
   ```
4. Open your browser (Streamlit will show the URL).

---

## 📂 Project Structure

* `engine_deepseek.py` — Main Streamlit app
* `index.json` — Dataset metadata index (required)
* `data\` — Data OSD
---

## ⚠️ Notes & Limitations

> Due to **limited resources and time**, this project was tested on **very small datasets**.
> It still works surprisingly well — but don’t expect production-level scaling yet. 😅

---

## 🙌 Contribution

PRs are welcome! You can:

* Improve dataset indexing (support more file types).
* Optimize memory usage for large datasets.

---

## 📜 License

MIT License — free to use, modify, and share.

---
