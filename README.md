# 🍳 MultiModal-Chef-RAG

**MultiModal-Chef-RAG** is an intelligent cooking assistant that bridges the gap between static recipes and visual demonstrations. By leveraging a RAG (Retrieval-Augmented Generation) workflow, the system retrieves recipes from **RecipeNLG** dataset and synchronizes them with precise instructional video segments from **YouCookII** dataset.

[Features](#-key-features) • [Setup & Installation](#-setup--installation) • [Demo Video](#-demo-video) • [RAG Workflow](#rag-workflow)

---

## ✨ Key Features
* **Dual-Stream Retrieval:** Seamlessly connects text-based recipes with corresponding video instruction clips.
* **Smart Video Sync:** Automatically identifies and plays the specific video segments matching the generated cooking steps.
* **Interactive UI:** A comprehensive Gradio interface featuring a chat module, synchronized video player, and a system inspection panel.
* **Resource Monitoring:** Real-time tracking of token usage and system performance.

---

## 🚀 Setup & Installation

### 1. Clone the Repository
Clone the repository and cd into the MultiModal-Chef-RAG folder.
```bash
cd MultiModal-Chef-RAG
```

### 2. Environment Setup
It is recommended to use a virtual environment (Python 3.9+).
```bash
pip install -r requirements.txt
```

### 3. Dataset Preparation
Download the datasets and place them exactly in the following directory structure:
| Dataset | Source | Target Path |
| :--- | :--- | :--- |
| **RecipeNLG** | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/recipenlg) | `project_data/full_dataset.csv` |
| **YouCookII** | [YouCook2](http://youcook2.eecs.umich.edu) | `project_data/YouCookII_downscaled/` |

### 4. Preprocessing
You must generate the FAISS indexes and embeddings before running the application for the first time:

```bash
# Generate recipe metadata embeddings
python -m preprocessing.metadata_preprocessing
# Generate video clip embeddings
python -m preprocessing.video_preprocessing
```
### 5. How to Start

Run one of the following commands:

**Option A: With UI (Gradio Interface)**
```bash
python -m main
```

**Option B: Without UI (CLI Pipeline)**
```bash
python -m pipeline
```

## 🎥 Demo Video
<div align="center">
  <video src="https://github.com/user-attachments/assets/c14e2c1d-e8c3-4520-a4f5-71256b224d11" width="100%" controls muted autoplay loop>
    Your browser does not support the video tag.
  </video>
</div>


<a id="rag-workflow"></a>
## 🏗️ RAG Workflow

1. **User Query:** User inputs a dish or craving.
2. **Recipe Retrieval:** `RecipeRetriever` performs a FAISS-based similarity search to find the best-matching recipe.
3. **LLM Generation:** The chatbot processes metadata to generate clear, structured instructions.
4. **Video Retrieval:** `VideoRetriever` matches the generated steps to specific timestamped clips.
5. **Augmentation:** The UI renders the instructions alongside a player synced to the retrieved clips.
