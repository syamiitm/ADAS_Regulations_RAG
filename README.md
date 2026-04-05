
# Multimodal RAG API: ADAS Regulatory Intelligence

Domain: AIS-162 & UNECE R131 ComplianceThis repository contains a Multimodal Retrieval-Augmented Generation (RAG) system designed for vehicle type approval and regulatory engineering. It allows users to query dense technical PDF documents containing text, tables, and diagrams.

# 1. Problem Statement

1.1 Domain Identification

The system targets vehicle homologation for Advanced Driver Assistance Systems (ADAS). Specifically, it indexes:AIS-162: Indian standard for Advanced Emergency Braking Systems (AEBS).UNECE R131: Uniform provisions for AEBS under the UN framework.

1.2 The Bottleneck

Homologation engineers currently perform manual side-by-side reviews of PDF packs. The challenges include:Tabular Complexity: Speed bands and timing thresholds are buried in dense tables.Visual Context: Scenario geometry (vehicle lanes, targets) is often defined only in figures, not text.Audit Risk: A single misread threshold ($km/h$) can invalidate a test program.

1.3 The RAG Solution

Unlike fine-tuning, this RAG approach ensures:Provenance: Every answer cites a specific filename, page, and chunk type.Dynamic Updates: New regulations can be indexed instantly without retraining.Multimodal Fusion: Uses a Vision Language Model (VLM) to convert diagrams into searchable text descriptions.

# 2. Architecture Overview

The pipeline handles PDF ingestion through a multi-stage process to preserve the semantic meaning of different document elements.

Code snippet

flowchart TB

    subgraph ingest [Ingestion Pipeline]
        PDF[PDF Upload] --> PM[PyMuPDF Extraction]
        PM --> T[Text Chunks]
        PM --> TB[Table Chunks - Markdown]
        PM --> IM[Images - PNG + VLM Summary]
        T --> E[Embedding Model]
        TB --> E
        IM --> E
        E --> FAISS[(FAISS Vector Store)]
    end

    subgraph query [Query Pipeline]
        Q[User Question] --> QE[Query Embedding]
        QE --> SR[FAISS Similarity Search]
        SR --> CTX[Context Assembly + Citations]
        CTX --> LLM[Grounded Chat LLM]
        LLM --> A[Answer + Sources JSON]
    end

# 3. Technology Stack

| Layer | Component | Rationale |
| :--- | :--- | :--- |
| **Document Parser** | **PyMuPDF (`fitz`)** | Fast, reliable extraction of text, tables, and image XREFs. |
| **Embeddings** | **NVIDIA Nemotron** | 2048-dimensional vectors for high-precision technical retrieval. |
| **Vector Store** | **FAISS** | In-memory, high-performance vector similarity search. |
| **VLM (Images)** | **Gemini 2.0 Flash** | Converts ADAS diagrams/figures into descriptive text before embedding. |
| **API Framework** | **FastAPI** | Provides automated OpenAPI/Swagger documentation at `/docs`. |



# 4. API Reference
GET /health

Returns system status, model readiness, and the current size of the vector index.

POST /ingest

Input: multipart/form-data (PDF file).

Action: Parses the PDF, generates VLM summaries for images, and updates the FAISS index.

POST /query

Input: JSON object with question and optional top_k.

Output: A grounded answer strictly based on the retrieved context, including detailed citations.


# 5. Setup & Execution
Prerequisites
Python 3.11+

API Keys for OpenRouter or OpenAI kept in .env

## Installtion

# Clone the repository

git clone <repo-url>
cd <repo-folder>

# Setup virtual environment

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies

pip install -r requirements.txt

# Configure environment

cp .env



# running the application

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Once running, access the Swagger UI at: http://localhost:8000/docs

# 6. Implementation Evidence

The following scenarios (found in the /screenshots folder) demonstrate the system's performance:

Server_Response.png: 
Successful server response

ingest_pdf1.png & Indest_pdf2.png: 
Successful ingestion of AIS-162 & r131 showing specific chunk counts for text, tables and images.

Query.png: 
A query response showing grounded text with page-level citations.

Answer_to_Query.png: 
A comparative query surfacing data from both AIS and UNECE documents simultaneously.

# 7. Limitations & Future Work

Table Extraction: 
Currently optimized for vector-based PDFs; OCR-based table detection is a planned upgrade.

Persistence: 
The current FAISS index is in-memory. Future versions will implement persistent disk storage for vector data.

Reranking: 
Integration of a Cross-Encoder for re-ranking search results to further improve precision.

# Academic Note:
 
This project was developed for the BITS PILANI WILP program as part of the Multimodal AI Assignment.

