# ADAS_Regulations_RAG
Retrieval‑Augmented Generation pipeline for comparing ADAS regulation standards

## Problem Statement

**Domain.** This work sits at the intersection of **vehicle type approval**, **Advanced Driver Assistance Systems (ADAS)**, and **regulatory engineering** for markets that reference both **India’s Automotive Industry Standard AIS‑162** (Advanced Emergency Braking Systems, AEBS, for commercial vehicle categories) and **UNECE Regulation No. 131** (uniform provisions for AEBS approval under the UN type‑approval framework). Homologation engineers, supplier compliance teams, and test-house reviewers routinely work from **statutory PDF packs** that interleave normative clauses, **annex tables of speed bands and performance limits**, and **diagrams of longitudinal collision‑threat scenarios**. The practical task is not “read the PDF once,” but to **cross‑walk** two instruments while keeping every answer traceable to the correct annex row, footnote, or figure note.

**Problem.** In day‑to‑day approval work, the bottleneck is **synthesising comparable intent across regimes** without misreading **tabular thresholds** or stripping **figure‑bound context**. AIS‑162 and R131 are written as **self‑contained approval texts**: limits appear in dense tables (e.g. scenario identifiers paired with maximum speeds, minimum required decelerations or warning timing, and exclusions), while **scenario geometry**—who moves, who brakes, lane relationship, and target class—is carried in **diagrams and defined terms** that the tables assume but do not repeat. Keyword search across a flat text export fails because (a) **table cells lose row/column semantics**, (b) **captions and figure numbering** drift away from the body text in extracted content, and (c) **cross‑references** (“see Annex …, figure …”) span modalities. Teams still default to manual side‑by‑side comparison, which is slow, error‑prone under deadline, and hard to audit when an interpretation is challenged.

**Why this is not a generic document Q&A task.** The user’s questions are **regulation‑shaped**: they ask for **commonalities** (e.g. shared scenario families, equivalent notions of collision threat, warning‑then‑brake logic, documentation of system boundaries and driver override), **divergences in technical requirements** (e.g. how each text scopes vehicle categories, phases applicability, prescribes test conditions, or structures approval evidence), and **reading aids** that tie a verbal requirement to **the exact table row** or **the diagram that defines the manoeuvre**. Answers must respect **specialised vocabulary** (vehicle categories M/N, approval vs. certification, system states, suppression conditions) and must not collapse two regimes into a vague “both require AEBS” summary. **Misalignment on a single speed band or scenario ID** can invalidate a test programme or a supplier declaration.

**Why retrieval‑augmented generation.** Fine‑tuning a model on regulations risks **stale law** and opaque updates; pure keyword search cannot reliably fuse **table**, **text**, and **diagram‑derived** evidence. A **RAG** pipeline keeps the **current PDF corpus** as the source of truth: chunks preserve modality and location (page, annex, chunk type), retrieval surfaces **the clause, the table fragment, and the image or diagram summary** that jointly answer a query, and the generator is constrained to **ground** its comparison in those retrieved spans. That supports **auditable** homologation reasoning—critical when a Notified Body or internal compliance review asks “which paragraph and which figure?”

**Expected outcomes.** The system should answer, with **explicit source references**, questions such as: where AIS‑162 and R131 **align on scenario philosophy and performance articulation**; where they **differ in technical prescription** (scope, thresholds, test boundaries, or approval artefacts); and how to **interpret a given table** in light of **its related diagram** (e.g. which scenario sketch corresponds to which annex table entries). Successful operation would **shorten cross‑regime gap analyses**, reduce misreads of **tabular limits**, and give engineers a **defensible, citation‑backed** basis for decisions on test scope and documentation structure—without replacing formal legal review.

---

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Entry point |
| `src/ingestion/` | Load and chunk documents, embeddings, index writes |
| `src/retrieval/` | Search, reranking, context assembly |
| `src/models/` | Schemas, LLM wrappers, prompts |
| `src/api/` | HTTP API (e.g. FastAPI) |
| `sample_documents/` | Example files for local testing |
| `screenshots/` | Screenshots for docs or demos |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Run

```bash
python main.py
```

## Environment

Copy `.env.example` to `.env` and set your keys and options.

**Defaults:** embeddings use OpenRouter **`nvidia/llama-nemotron-embed-vl-1b-v2:free`** (FAISS dimension **2048**) via `CHAT_BASE_URL` + `OPENROUTER_API_KEY`; figure captions use **`OPENAI_API_KEY`** (VLM). Override with `EMBEDDING_MODEL=text-embedding-3-small` and `EMBEDDING_DIMENSION=1536` for OpenAI-only embeddings.

**Important:** If you ingested with a **1536-d** index, switch to **2048** (or change model) only after **restarting** the app and **re-ingesting** PDFs so FAISS matches the embedding size.
