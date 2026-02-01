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
Raw Data (Resumes)
↓
Text Cleaning & Structuring
↓
Sentence Embeddings (MiniLM)
↓
Vector Storage (Endee)
↓
Semantic Search (Cosine Similarity)
↓
Ranked Resume Results based on job description
```

---

### Data Preparation

**Datasets Used:**
- Resume Dataset (Kaggle – Snehaan Bhawal)

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
Similarity scores typically range between **0.5–0.9**, which is expected for general-purpose embeddings and diverse datasets. The system prioritizes **relative ranking**, not absolute similarity.

---

## Testing & Validation

The system was validated using multiple test scenarios:

### Test Case 1: Chef
- Similarity scores: ~0.85–0.87
  <img width="917" height="620" alt="image" src="https://github.com/user-attachments/assets/679a1551-4cb6-45f9-8036-aa26617ea476" />
  <img width="746" height="213" alt="image" src="https://github.com/user-attachments/assets/33152c12-295d-4f4f-9db8-b0baf776d6b0" />



### Test Case 2: Sales Executive
- Similarity scores: ~0.25–0.40
  <img width="961" height="604" alt="image" src="https://github.com/user-attachments/assets/fed2b91a-56bd-4619-9e53-c7035a9057dc" />
  <img width="937" height="211" alt="image" src="https://github.com/user-attachments/assets/138d3ece-2edf-4c78-aef7-249be42188d0" />



### Test Case 3: HR Manager
- Similarity scores: ~0.75-0.78
  <img width="1101" height="684" alt="image" src="https://github.com/user-attachments/assets/4295b5b2-e162-4d52-8c14-b56a79dfa17a" />
  <img width="808" height="256" alt="image" src="https://github.com/user-attachments/assets/d0a6ba23-cdec-4c92-ae65-325c61001108" />


### Test Case 4: Financial Analyst
- Similarity scores: ~0.65-0.70
  <img width="1008" height="652" alt="image" src="https://github.com/user-attachments/assets/a755e25c-1011-4092-b4ea-8ad46e4d2bd4" />
  <img width="936" height="209" alt="image" src="https://github.com/user-attachments/assets/8fac470e-786e-4b82-8d1f-57a1c94247da" />


### Test Case 5: Financial Analyst
- Similarity scores: ~0.66-0.71
  <img width="932" height="616" alt="image" src="https://github.com/user-attachments/assets/ec76c099-12fa-4906-a458-727139b104f5" />
  <img width="809" height="208" alt="image" src="https://github.com/user-attachments/assets/4b405be5-7156-4bcf-82da-67ce57b0063f" />


These tests validate that the system correctly captures **semantic relevance**.

---

## Project Structure

```
Ai_ml_project/
│
├── data/
│   └── resumes.csv
│
├── outputs/
│   └── clean_resumes.csv
│
├── embeddings/
│   ├── resume_embeddings.npy
│   └── resume_metadata.csv
│
├── src/
│   ├── utils.py
│   ├── preprocess_resumes.py
│   ├── embedder.py
│   ├── generate_embeddings.py
│   ├── store_in_endee.py
│   └── match_resumes.py
│
└── README.md
```

---

## Setup & Execution Instructions

### Prerequisites
- Python 3.9+
- Virtual environment recommended
- Endee server running locally (OSS)

---

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
pip install sentence-transformers endee pandas numpy
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

---

## Conclusion

This project demonstrates a production-grade semantic search pipeline using embeddings and Endee. It replaces brittle keyword-based matching with a scalable, context-aware, and industry-aligned resume screening system.

---

## Author

**Kurakula Harshitha**  
AI / ML Project – Resume–Job Matching using Endee
