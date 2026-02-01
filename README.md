# 📄 Resume–Job Matching using Semantic Search & Endee Vector Database

## Project Overview

Hiring platforms and recruiters often rely on keyword-based systems to match resumes with job descriptions. These systems fail to capture **semantic meaning**, leading to poor matching results when different wording is used for similar skills or roles.

This project implements an **AI-powered Resume–Job Matching system** using **semantic search** and **vector similarity**. It converts resumes and job descriptions into dense vector embeddings and uses **Endee**, a high-performance open-source vector database, to retrieve and rank the most relevant resumes for a given job description.

The system focuses on **meaning-based matching**, ensuring that resumes relevant in context—not just keywords—are ranked higher.

---

## Problem Statement

Traditional resume screening systems suffer from the following limitations:

- Keyword dependency (fails with synonyms or paraphrased skills)
- Poor understanding of context and role relevance
- Inability to rank resumes meaningfully

### Objective

To design and implement a system that:
- Represents resumes and job descriptions as semantic vectors
- Stores and searches vectors efficiently using Endee
- Ranks resumes based on semantic similarity to a job description

---

## System Design & Technical Approach

### High-Level Architecture

```
Raw Data (Resumes & Jobs)
↓
Text Cleaning & Structuring
↓
Sentence Embeddings (MiniLM)
↓
Vector Storage (Endee)
↓
Semantic Search (Cosine Similarity)
↓
Ranked Resume Results
```

---

### Data Preparation

**Datasets Used:**
- Resume Dataset (Kaggle – Snehaan Bhawal)
- Job Description Dataset (Kaggle – Kshitiz Regmi)

**Preprocessing Steps:**
- Remove null or empty records
- Clean text (lowercasing, whitespace normalization)
- Add semantic structure:
  - Role / category context
  - Skills and experience emphasis

This step is critical because **embedding quality heavily depends on text structure**.

---

### Embedding Generation

- Model used: `all-MiniLM-L6-v2` (Sentence Transformers)
- Same model used for:
  - Resumes
  - Job descriptions

**Why this model?**
- Lightweight and fast
- High-quality semantic representations
- Suitable for large-scale similarity search

**Important Considerations:**
- Embeddings are **normalized**
- Same embedding space ensures meaningful cosine similarity

---

### Vector Database: Endee

**Endee** is used as the core vector database for this project.

#### Why Endee?
- High-performance vector similarity search
- Open-source and self-hosted
- Optimized for large-scale embedding workloads
- Clean API for index and vector operations

#### How Endee is Used:
- A vector index (`resumes_index`) is created with cosine similarity
- Each resume embedding is stored with:
  - Unique resume ID
  - Vector embedding
  - Metadata (category, source)

Endee efficiently retrieves the **Top-K nearest resume vectors** for a given job embedding.

---

### Job Matching Logic

For a given job description:
1. Convert job text into an embedding
2. Query Endee using cosine similarity
3. Retrieve Top-K closest resume vectors
4. Rank resumes by similarity score

**Note:**  
Similarity scores typically range between **0.45–0.70**, which is expected for general-purpose embeddings and diverse datasets. The system prioritizes **relative ranking**, not absolute similarity.

---

## Testing & Validation

The system was validated using multiple test scenarios:

### Test Case 1: Technical Job (Flutter Developer)
- Technical resumes ranked higher
- Similarity scores: ~0.50–0.65

### Test Case 2: Management Job (HR Manager)
- Technical resumes ranked lower
- Similarity scores: ~0.25–0.40

### Test Case 3: Unrelated Domain (Finance)
- Very low similarity scores
- Confirms semantic separation

These tests validate that the system correctly captures **semantic relevance**.

---

## Project Structure

```
Ai_ml_project/
│
├── data/
│   ├── resumes.csv
│   └── jobs.csv
│
├── outputs/
│   ├── clean_resumes.csv
│   └── clean_jobs.csv
│
├── embeddings/
│   ├── resume_embeddings.npy
│   └── resume_metadata.csv
│
├── src/
│   ├── utils.py
│   ├── preprocess_resumes.py
│   ├── preprocess_jobs.py
│   ├── embedder.py
│   ├── generate_embeddings.py
│   ├── store_in_endee.py
│   └── match_resumes.py
│
├── README.md
└── requirements.txt
```

---

## Setup & Execution Instructions

### Prerequisites
- Python 3.9+
- Virtual environment recommended
- Endee server running locally (OSS)

---

### Clone Repository
```bash
git clone <your-github-repo-url>
cd Ai_ml_project
```

### Create Virtual Environment
```bash
python -m venv venv
```

**Activate:**

- **Linux / macOS**
  ```bash
  source venv/bin/activate
  ```

- **Windows**
  ```bash
  venv\Scripts\activate
  ```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Start Endee Server (Docker)
```bash
docker compose up -d
```

**Endee will be available at:**  
http://localhost:8080

### Run Data Preprocessing
```bash
python src/preprocess_resumes.py
python src/preprocess_jobs.py
```

### Generate Resume Embeddings
```bash
python src/generate_embeddings.py
```

### Store Embeddings in Endee
```bash
python src/store_in_endee.py
```

### Match Resumes with Job Description
```bash
python src/match_resumes.py
```

This command outputs the Top-K most relevant resumes ranked by semantic similarity.

---

## Future Improvements

- Fine-tune embeddings on resume–job pairs
- Add metadata-based filtering (experience, location)
- Support incremental updates to embeddings
- Build a web interface for recruiters
- Scale Endee for millions of resumes

---

## Conclusion

This project demonstrates a production-grade semantic search pipeline using embeddings and Endee. It replaces brittle keyword-based matching with a scalable, context-aware, and industry-aligned resume screening system.

---

## Author

**Harshitha**  
AI / ML Project – Resume–Job Matching using Endee