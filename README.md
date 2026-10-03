# ProjectGenome — Complete Implementation & Research Roadmap

**Project:** ProjectGenome  
**Research Title:** *Beyond Semantic Retrieval: Knowledge Graph-Aware Retrieval for Repository-Level Software Understanding*  
**Document Type:** Operational master roadmap  
**Status:** Finalized implementation roadmap  
**Primary KG Technology:** NetworkX  
**Main Generator:** Qwen2.5-Coder-7B-Instruct  
**Main Semantic Encoder:** CodeBERT

---

# 1. Purpose of This Roadmap

This document is the step-by-step implementation and research roadmap for ProjectGenome.

The purpose is to take the project from the current repository-analysis/KG foundation to a complete experimental system and finally to the research paper.

The roadmap is organized as **phases**. Each phase contains:

- objective
- tasks
- implementation details
- deliverables
- exit condition

Move to the next phase only after the current phase's exit condition is satisfied.

---

# 2. Research Question

> **Can Knowledge Graph-aware retrieval help an LLM understand and answer questions about a software repository more accurately and completely than conventional retrieval methods?**

The project is therefore not simply about constructing a Knowledge Graph.

The actual research pipeline is:

```text
Repository
    ↓
Repository Analysis
    ↓
Normalized Analyzer JSON
    ↓
Knowledge Graph
    ↓
Retrieval
    ↓
Retrieved Repository Context
    ↓
LLM
    ↓
Answer
    ↓
Evaluation
```

The research comparison focuses on whether adding repository structure and graph-based context expansion improves repository-level question answering.

---

# 3. Final Project Scope

## 3.1 Main Retrieval Methods

The core experiment contains five retrieval strategies:

1. **BM25**
2. **Semantic Retrieval**
3. **Structural Retrieval**
4. **Hybrid Retrieval**
5. **Graph-Aware Retrieval**

## 3.2 Main Models

### Semantic Retrieval

**CodeBERT**

### Answer Generation

**Qwen2.5-Coder-7B-Instruct**

## 3.3 Secondary Experiments

### GraphCodeBERT Ablation

Compare:

```text
Semantic + CodeBERT
Semantic + GraphCodeBERT

Graph-Aware + CodeBERT
Graph-Aware + GraphCodeBERT
```

All use Qwen for generation.

### Optional Generator Robustness

Only if resources and time allow:

```text
Semantic + Qwen
Semantic + DeepSeek-Coder-V2-Lite-Instruct
Semantic + StarCoder2-7B

Graph-Aware + Qwen
Graph-Aware + DeepSeek-Coder-V2-Lite-Instruct
Graph-Aware + StarCoder2-7B
```

These are **optional secondary experiments**, not requirements for the main result.

---

# 4. Final Main Experiment

| ID | Retrieval | Semantic Model | Generator |
|---|---|---|---|
| B1 | BM25 | — | Qwen |
| B2 | Semantic | CodeBERT | Qwen |
| B3 | Structural | — | Qwen |
| B4 | Hybrid | CodeBERT | Qwen |
| B5 | Graph-Aware | CodeBERT | Qwen |

If the verified benchmark contains 720 usable questions:

```text
5 retrieval systems × 720 questions
= 3,600 generations
```

The main experiment should be completed before running optional model comparisons.

---

# 5. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Repository parsing | Tree-sitter |
| Analyzer output | JSON |
| Knowledge Graph | NetworkX |
| Graph type | NetworkX MultiDiGraph |
| Keyword retrieval | BM25 |
| Main semantic model | CodeBERT |
| Ablation semantic model | GraphCodeBERT |
| Main generator | Qwen2.5-Coder-7B-Instruct |
| Optional generators | DeepSeek-Coder-V2-Lite-Instruct, StarCoder2-7B |
| Testing | pytest |
| Visualization | NetworkX + Matplotlib |
| Data analysis | pandas |

**Neo4j is not part of the current implementation.**

---

# 6. Architecture

```text
                    ┌─────────────────────┐
                    │     Repository      │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Repository Analysis │
                    │    / Tree-sitter    │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Analyzer JSON       │
                    │ normalized contract │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Knowledge Graph     │
                    │ NetworkX MultiDiGraph│
                    └──────────┬──────────┘
                               ↓
             ┌─────────────────┼─────────────────┐
             ↓                 ↓                 ↓
          BM25             Semantic          Structural
                              CodeBERT
             │                 │                 │
             └─────────────────┼─────────────────┘
                               ↓
                         Hybrid / Graph-
                         Aware Retrieval
                               ↓
                    ┌─────────────────────┐
                    │ Context Builder     │
                    └──────────┬──────────┘
                               ↓
                    ┌─────────────────────┐
                    │ Qwen2.5-Coder-7B    │
                    └──────────┬──────────┘
                               ↓
                           Answer
                               ↓
                    ┌─────────────────────┐
                    │ Evaluation          │
                    └─────────────────────┘
```

## 6.1 Subsystems Overview & Implementation Status

ProjectGenome is built as a modular software intelligence and graph retrieval pipeline:

### 1. Repository Intelligence Subsystem (`projectgenome/`)
- **Purpose:** Deterministic static analysis and AST structure extraction for code repositories.
- **Components:**
  - `scanner/repository_scanner.py`: Traverses directories, normalizes file paths, derives logical module names.
  - `parsers/python_parser.py`: AST parsing, class, function, method, import, inheritance, and call expression extraction.
  - `extractors/entity_extractor.py` & `relationship_extractor.py`: Constructs deterministic entity IDs (`type:path:name`), resolves calls statically, handles unresolved external calls, tracks line/column provenance.
  - `exporters/json_exporter.py`: Emits canonical JSON sorted by ID with SHA-256 canonical integrity hash.
- **CLI Usage:**
  ```bash
  python -m projectgenome.main --repo sample_repo --output sample_analysis.json
  ```
- **Tests:** `tests/test_models.py`, `tests/test_scanner.py`, `tests/test_parser_and_extractors.py` (11/11 passing).

### 2. Knowledge Graph Subsystem (`src/knowledge_graph/`)
- **Purpose:** NetworkX `MultiDiGraph` construction, graph schema validation, graph traversal, and neighborhood context extraction.
- **Components:**
  - `builder.py`: Ingests normalized JSON (`sample_analysis.json`) into NetworkX graph without Module nodes.
  - `graph.py` & `schema.py`: Typed graph nodes, edges, properties, and constraint enforcement.
  - `traversal.py` & `queries.py`: Multi-hop neighborhood expansion, callers/callees lookup, dependency path discovery.
  - `serialization.py` & `visualization/`: JSON/GraphML export & NetworkX visualizer.
- **Usage:**
  ```python
  from src.knowledge_graph.builder import build_from_analysis
  kg = build_from_analysis("sample_analysis.json")
  ```
- **Tests:** `tests/knowledge_graph/test_*.py` (39/39 passing).

### 3. Retrieval Subsystem (`src/retrieval/`)
- **Purpose:** Multi-strategy code retrieval supporting lexical, semantic, and structural baselines under a common retrieval interface (`BaseRetriever`, `RetrievedItem`).
- **Components:**
  - `base.py`: Common retrieval contract (`BaseRetriever`, `RetrievedItem`, `DummyRetriever`).
  - `chunker.py`: Canonical `CodeChunk` generation from repository analysis and Knowledge Graph entities.
  - `bm25.py`: Lexical baseline (`BM25Retriever`, B1).
  - `embeddings.py` & `index.py`: Dense 768-dimensional CodeBERT vector representations and in-memory vector index (`CodeVectorIndex`).
  - `semantic.py`: Dense semantic retrieval (`SemanticRetriever`, B2).
  - `structural.py`: Deterministic multi-hop Knowledge Graph traversal retriever (`StructuralRetriever`, B3).
- **Status:** B1 (BM25), B2 (Semantic CodeBERT), and B3 (Structural) implemented and verified. B4 (Hybrid) and B5 (Graph-Aware) pending.
- **Tests:** `tests/retrieval/test_*.py` (111/111 passing; 162/162 repository-wide passing).

---

# PHASE 0 — Freeze the Project Scope

## Objective

Remove ambiguity before implementation expands.

## 0.1 Freeze the Research Question

Use the research question in Section 2.

## 0.2 Freeze the Main Retrieval Methods

The main comparison is:

```text
BM25
Semantic
Structural
Hybrid
Graph-Aware
```

## 0.3 Freeze the Main Model Pair

The primary controlled experiment uses:

```text
CodeBERT
    +
Qwen2.5-Coder-7B-Instruct
```

## 0.4 Freeze Evaluation

Retrieval:

- Precision@K
- Recall@K
- MRR

Answers:

- Correctness
- Completeness
- Faithfulness
- Context efficiency

## 0.5 Define Controlled Variables

Keep the following fixed wherever possible:

- benchmark questions
- repositories
- retrieved-context budget
- top-K definition
- prompt format
- generator settings
- evaluation protocol

## Deliverable

A short frozen experiment configuration.

## Exit Condition

Every team member knows exactly what constitutes the main experiment.

---

# PHASE 1 — Environment and Project Foundation

## Objective

Ensure the project can be developed and tested reproducibly.

## 1.1 Python Environment

Use the existing Python environment:

```text
Python 3.12.2
```

Create/use:

```text
.venv/
```

## 1.2 Dependencies

Current project dependencies include:

```text
tree-sitter
tree-sitter-language-pack
networkx
GitPython
pydantic
pytest
pandas
```

Retrieval/model dependencies should be added only when those phases begin.

## 1.3 Project Structure

Target structure:

```text
ProjectGenome/
│
├── data/
│   ├── sample/
│   ├── generated/
│   └── benchmark/
│
├── docs/
├── examples/
│
├── src/
│   ├── repository_analysis/
│   ├── knowledge_graph/
│   ├── retrieval/
│   ├── context/
│   ├── generation/
│   └── evaluation/
│
├── tests/
├── requirements.txt
└── README.md
```

## 1.4 Baseline Test

Run:

```bash
.venv/bin/pytest -v
```

## Deliverable

Working development environment and stable project structure.

## Exit Condition

Tests execute successfully and the project can be imported without environment errors.

---

# PHASE 2 — Freeze the Repository Analyzer Contract

## Objective

Define exactly what enters the Knowledge Graph.

The repository analyzer is already implemented. Do not redesign it during the KG work.

## 2.1 Entity Types

The canonical analyzer entities are:

```text
Repository
Directory
File
Class
Function
Method
```

There is **no separate Module entity** for ordinary Python files.

A logical module name can remain metadata on a File.

## 2.2 Primitive Relationships

```text
CONTAINS
IMPORTS
INHERITS
CALLS
```

## 2.3 Derived Relationship

```text
DEPENDS_ON
```

`DEPENDS_ON` is derived from primitive evidence.

## 2.4 Deterministic Information

Preserve:

- deterministic IDs
- source locations
- provenance
- metadata
- relationship information

## 2.5 Unresolved References

Unresolved calls must remain unresolved.

For example:

```text
target = unresolved:print
resolved = false
```

The KG must not invent repository entities for unresolved external calls.

Synthetic placeholder nodes may be created when useful, but they must be clearly marked:

```text
synthetic = true
unresolved = true
```

## 2.6 Canonical Fixture

Use:

```text
data/sample/sample_analysis.json
```

as the canonical KG input fixture.

## Deliverable

Stable analyzer-to-KG contract.

## Exit Condition

The KG can consume analyzer JSON without depending on the analyzer's internal implementation.

---

# PHASE 3 — Implement the Knowledge Graph Core

## Objective

Convert normalized analyzer JSON into a reliable NetworkX Knowledge Graph.

## 3.1 Schema

Implement/enforce:

```text
NodeType
RelationshipType
```

## 3.2 Data Models

Create validated models for:

```text
Node
Relationship
Graph input/output
```

Validate:

- IDs
- names
- types
- source
- target
- metadata

## 3.3 Graph Representation

Use:

```python
networkx.MultiDiGraph
```

This allows multiple relationship types between the same pair of entities.

## 3.4 Graph Builder

Implement:

```text
Analyzer JSON
      ↓
Input validation
      ↓
Node creation
      ↓
Primitive relationship creation
      ↓
Derived DEPENDS_ON creation
      ↓
Graph validation
      ↓
NetworkX graph
```

## 3.5 Graph Access Layer

Implement operations such as:

```text
get_node()
get_nodes_by_type()
get_relationships()
get_neighbors()
get_repository()
```

## 3.6 Serialization

Support:

```text
JSON
GraphML
```

## Deliverables

```text
src/knowledge_graph/
├── schema.py
├── models.py
├── graph.py
├── builder.py
├── queries.py
├── traversal.py
├── validation.py
└── serialization.py
```

## Exit Condition

`sample_analysis.json` can be converted into a valid NetworkX graph and serialized successfully.

---

# PHASE 4 — Knowledge Graph Testing and Validation

## Objective

Make sure the KG is reliable before retrieval depends on it.

## 4.1 Unit Tests

Test:

- valid nodes
- invalid nodes
- invalid IDs
- duplicate nodes
- invalid types
- invalid relationships
- missing source
- missing target
- metadata

## 4.2 Relationship Tests

Verify:

```text
CONTAINS
IMPORTS
INHERITS
CALLS
DEPENDS_ON
```

## 4.3 Unresolved Reference Tests

Verify:

- unresolved references are preserved
- unresolved nodes are marked
- unresolved references are not treated as repository entities

## 4.4 Integration Test

Run:

```text
sample_analysis.json
        ↓
Builder
        ↓
NetworkX graph
        ↓
Validation
        ↓
Serialization
```

## 4.5 Current Baseline

The current KG implementation has:

```text
39 tests
39 passed
0 failures
```

This should remain the baseline after future changes.

## Exit Condition

All KG tests pass before retrieval implementation begins.

---

# PHASE 5 — Knowledge Graph Visualization

## Objective

Create a visual debugging and inspection tool.

Visualization is not itself a research variable.

## 5.1 Generic Input

The visualizer must load:

```text
repository_graph.json
```

rather than hardcoding the current sample.

## 5.2 Node Visualization

Distinguish:

```text
Repository
Directory
File
Class
Function
Method
Unresolved/Synthetic
```

## 5.3 Relationship Visualization

Distinguish:

```text
CONTAINS
IMPORTS
CALLS
INHERITS
DEPENDS_ON
```

## 5.4 Required Features

Include:

- readable labels
- relationship direction
- legend
- clear hierarchy
- unresolved-node distinction

## 5.5 Output

Generate:

```text
data/generated/repository_graph.png
```

## Exit Condition

The same visualization code works for different repository graphs without changing node names or coordinates manually.

---

# PHASE 6 — Prepare the Benchmark and Repository Corpus

## Objective

Prepare the actual research data.

## 6.1 Verify SWE-QA

Use:

```text
SWE-QA-Benchmark
```

Verify before implementation:

- exact number of usable questions
- repository identifiers
- question format
- answer format
- repository availability
- evidence/ground truth availability
- dataset split
- license

Do not assume the benchmark structure without checking the actual release.

## 6.2 Create Question Mapping

Each benchmark item should resolve to:

```text
Question
Repository
Ground-truth answer
Ground-truth evidence
```

## 6.3 Prepare Repositories

For every repository:

```text
Repository
    ↓
Repository Analyzer
    ↓
Analyzer JSON
    ↓
Knowledge Graph
```

## 6.4 Prepare Retrieval Units

Repository content should be represented using units such as:

```text
File
Class
Function
Method
```

Each retrieval unit should retain:

```text
repository_id
entity_id
file_path
entity_type
source location
source code
```

## 6.5 Create Ground-Truth Evidence

Where the benchmark supports it, identify the repository code entities/chunks that contain evidence required to answer each question.

This is necessary for retrieval evaluation.

## Deliverable

A benchmark/corpus mapping that connects:

```text
Question → Repository → Ground-truth evidence
```

## Exit Condition

A test question can be traced from the benchmark to the exact repository that must be searched.

---

# PHASE 7 — Create the Common Retrieval Interface

## Objective

Make all retrieval systems comparable.

## 7.1 Common Input

Every retriever receives:

```text
question
repository
k
```

## 7.2 Common Output

Every retriever returns items containing:

```text
entity_id
file_path
source_code
score
retrieval_method
provenance
```

## 7.3 Common Retrieval Function

Conceptually:

```python
retrieve(
    question,
    repository,
    k
)
```

## 7.4 Common Rules

All retrieval methods must use:

- same repository
- same question
- same K
- same retrieval-unit definition
- same output format

## Deliverable

A retrieval interface that supports plug-in retrievers.

## Exit Condition

A dummy retriever can pass through the entire retrieval → context pipeline.

---

# PHASE 8 — Implement BM25 Retrieval

## Objective

Create the lexical baseline.

## 8.1 Build Index

Index the repository retrieval units.

Possible units:

```text
files
classes
functions
methods
```

## 8.2 Query Processing

Input:

```text
natural-language question
```

BM25 produces ranked repository units.

## 8.3 Return Top-K

Return:

```text
Top-K code units
```

with scores and provenance.

## 8.4 Test

Check:

- relevant terms retrieve expected code
- ranking is deterministic
- metadata is preserved
- K is respected

## Deliverable

```text
BM25Retriever
```

## Exit Condition

BM25 can retrieve top-K evidence for a benchmark question.

---

# PHASE 9 — Implement Semantic Retrieval with CodeBERT

## Objective

Implement the primary semantic retrieval baseline.

## 9.1 Code Representation

Encode repository retrieval units using CodeBERT.

## 9.2 Query Representation

Encode the natural-language question using the same representation pipeline.

## 9.3 Similarity

Calculate semantic similarity between:

```text
question embedding
        ↕
code embedding
```

## 9.4 Ranking

Sort repository units by similarity.

## 9.5 Return Top-K

Return:

```text
entity
file
source
similarity score
provenance
```

## 9.6 Cache Embeddings

Repository embeddings should be computed once and reused.

Question embeddings can be generated during evaluation.

## Deliverable

```text
SemanticRetriever(CodeBERT)
```

## Exit Condition

The same question can be retrieved using semantic similarity rather than lexical matching.

---

# PHASE 10 — Implement Structural Retrieval

## Objective

Retrieve repository context using explicit Knowledge Graph relationships.

Structural retrieval is different from semantic retrieval.

## 10.1 Identify Candidate Entities

Map the question to relevant repository entities using deterministic signals available to the structural retriever.

Possible signals include:

- explicit entity names
- file names
- class names
- function names
- method names
- identifiers mentioned in the question

## 10.2 Query the KG

Use relationships:

```text
CONTAINS
IMPORTS
INHERITS
CALLS
DEPENDS_ON
```

## 10.3 Traverse

Retrieve connected entities.

Example:

```text
Function A
   ↓ CALLS
Function B
   ↓ CONTAINS
File B
```

## 10.4 Convert to Context

Retrieve source code associated with selected entities.

## 10.5 Preserve Graph Evidence

Record:

```text
starting entity
relationship path
target entity
```

## Deliverable

```text
StructuralRetriever
```

## Exit Condition

Structural retrieval can produce repository context using KG relationships without CodeBERT.

---

# PHASE 11 — Implement Hybrid Retrieval

## Objective

Combine semantic relevance and repository structure.

## 11.1 Inputs

Use:

```text
CodeBERT semantic results
+
Structural KG results
```

## 11.2 Normalize Scores

Because semantic and structural scores may use different scales, define a deterministic normalization method.

## 11.3 Combine

Use a fixed combination rule.

Conceptually:

```text
Hybrid Score
=
α × Semantic Score
+
β × Structural Score
```

The exact values of α and β must be fixed before the main experiment.

## 11.4 Rank

Produce one combined ranking.

## 11.5 Preserve Evidence

The output should indicate whether an item was supported by:

```text
semantic evidence
structural evidence
both
```

## Deliverable

```text
HybridRetriever
```

## Exit Condition

Hybrid retrieval produces one deterministic ranking from semantic and structural evidence.

---

# PHASE 12 — Implement Graph-Aware Retrieval

## Objective

This is the **proposed retrieval method** and the central research component.

The key idea is:

> Semantic retrieval finds relevant starting points; the Knowledge Graph then expands the retrieved context using repository relationships.

## 12.1 Step 1 — Semantic Seed Retrieval

Use CodeBERT:

```text
Question
   ↓
CodeBERT
   ↓
Initial Top-K entities
```

These are the seed entities.

## 12.2 Step 2 — Map Seeds to KG

For every semantic result:

```text
retrieved code unit
      ↓
entity_id
      ↓
Knowledge Graph node
```

## 12.3 Step 3 — Traverse the KG

Expand from seed entities using relevant relationships:

```text
CONTAINS
IMPORTS
INHERITS
CALLS
DEPENDS_ON
```

## 12.4 Step 4 — Control Traversal

Fix:

- seed K
- traversal depth
- allowed relationship types
- maximum expanded nodes
- final context K/token budget

These must not change between questions unless explicitly defined by the algorithm.

## 12.5 Step 5 — Rank Expanded Entities

Rank candidate context using a deterministic rule incorporating:

- seed relevance
- graph distance
- relationship type
- semantic score

## 12.6 Step 6 — Build Final Context

The final context contains the most useful semantic seeds plus structurally connected evidence.

## 12.7 Preserve Graph Paths

For every graph-expanded item, preserve:

```text
seed
→ relationship
→ intermediate entity
→ relationship
→ target
```

This is important for explaining why the graph-aware retriever selected the context.

## Deliverable

```text
GraphAwareRetriever
```

## Exit Condition

Graph-aware retrieval performs:

```text
Question
  ↓
Semantic seeds
  ↓
KG mapping
  ↓
Graph traversal
  ↓
Expanded candidates
  ↓
Ranking
  ↓
Final context
```

without simply becoming another semantic retriever.

---

# PHASE 13 — Build the Common Context Builder

## Objective

Ensure that retrieval methods are compared fairly after retrieval.

All five retrievers must feed into the same context-building stage.

## 13.1 Input

The context builder receives:

```text
retrieved items
```

## 13.2 Include

For each item where applicable:

```text
file path
entity name
entity type
source code
retrieval score
relationship evidence
```

## 13.3 Context Budget

Set one context/token budget for the main experiment.

Do not allow one retrieval method to provide dramatically more context than another.

## 13.4 Prompt Structure

Use the same general prompt format for every retrieval method.

Example:

```text
You are answering a question about a software repository.

Question:
{question}

Repository Context:
{retrieved_context}

Answer the question using the provided repository context.
```

## 13.5 No Extra Information

The context builder must not introduce repository information that was not retrieved.

## Deliverable

```text
ContextBuilder
```

## Exit Condition

Every retrieval method produces compatible LLM input.

---

# PHASE 14 — Implement Qwen Generation Pipeline

## Objective

Connect retrieval to answer generation.

## 14.1 Input

```text
Question
+
Retrieved Context
```

## 14.2 Generator

Use:

```text
Qwen2.5-Coder-7B-Instruct
```

## 14.3 Controlled Settings

Record:

- model version
- temperature
- max output tokens
- context size
- prompt template
- generation parameters

Use fixed settings for the main comparison.

## 14.4 Save Every Result

Store:

```text
question_id
repository_id
retrieval_method
retrieved_entities
context
model
generation settings
answer
latency
```

## Deliverable

End-to-end:

```text
Question
→ Retrieval
→ Context
→ Qwen
→ Answer
```

## Exit Condition

One benchmark question can be run automatically through every main retrieval system.

---

# PHASE 15 — Implement Retrieval Evaluation

## Objective

Measure whether each retrieval method actually retrieves relevant evidence.

This is separate from answer quality.

## 15.1 Precision@K

Measures:

> How much of the retrieved top-K evidence is relevant?

## 15.2 Recall@K

Measures:

> How much of the required evidence was retrieved?

## 15.3 MRR

Measures:

> How high does the first relevant result appear?

## 15.4 Evaluate Every Main Method

Run:

```text
BM25
Semantic
Structural
Hybrid
Graph-Aware
```

## Deliverable

Retrieval metrics per question and aggregated by method.

## Exit Condition

A retrieval-results file exists for every main method.

---

# PHASE 16 — Implement Answer Evaluation

## Objective

Determine whether better retrieval actually produces better answers.

## 16.1 Correctness

Does the answer correctly answer the question?

## 16.2 Completeness

Does it cover the important aspects required by the ground truth?

## 16.3 Faithfulness

Is the answer supported by the retrieved repository context?

## 16.4 Context Efficiency

Measure useful evidence relative to the amount of retrieved context.

Possible measurements include:

```text
relevant evidence / retrieved evidence
```

and/or token efficiency.

The exact operational definition must be fixed before the main experiment.

## Deliverable

Per-question answer evaluation.

## Exit Condition

Every generated answer can be evaluated consistently.

---

# PHASE 17 — Build the Pilot Experiment

## Objective

Find pipeline problems before running the full benchmark.

## 17.1 Pilot Size

Use approximately:

```text
5 repositories
×
10 questions per repository
=
50 questions
```

## 17.2 Run All Main Systems

```text
50 questions
×
5 retrieval methods
=
250 generations
```

## 17.3 Check

### Dataset

- every question maps correctly
- repositories are available
- ground truth is readable

### Retrieval

- every retriever returns results
- K is correct
- provenance is preserved

### KG

- repository graphs load
- entity mapping works
- traversal works

### Generation

- prompts are valid
- Qwen receives context
- answers are saved

### Evaluation

- retrieval metrics compute
- answer evaluation works
- result files are complete

## Deliverable

Pilot report.

## Exit Condition

No pipeline-breaking errors remain.

---

# PHASE 18 — Run the Main Experiment

## Objective

Run the complete controlled experiment.

## 18.1 Main Systems

```text
B1 BM25
B2 Semantic + CodeBERT
B3 Structural
B4 Hybrid + CodeBERT
B5 Graph-Aware + CodeBERT
```

## 18.2 Full Benchmark

If 720 questions are confirmed:

```text
720 × 5 = 3,600 generations
```

## 18.3 Automated Execution

Create a runner that:

1. loads benchmark
2. selects repository
3. loads repository data/KG
4. runs retriever
5. builds context
6. runs Qwen
7. stores answer
8. stores retrieval results
9. records latency
10. continues to next question

## 18.4 Failure Handling

If a question fails:

```text
save error
save question ID
continue experiment
```

Do not silently discard failed cases.

## Deliverable

Complete raw experiment results.

## Exit Condition

All usable benchmark questions have results for all five systems or are explicitly documented as failed/excluded.

---

# PHASE 19 — GraphCodeBERT Ablation

## Objective

Determine whether the graph-aware improvement depends specifically on CodeBERT.

## 19.1 Run

```text
Semantic + CodeBERT
Semantic + GraphCodeBERT

Graph-Aware + CodeBERT
Graph-Aware + GraphCodeBERT
```

## 19.2 Keep Everything Else Fixed

Keep constant:

- questions
- repositories
- K
- context budget
- graph traversal
- generator
- prompt
- evaluation

Only change the semantic representation model.

## Deliverable

Embedding ablation results.

## Exit Condition

The effect of changing CodeBERT → GraphCodeBERT can be analyzed independently.

---

# PHASE 20 — Optional Generator Robustness Experiment

## Objective

Check whether the retrieval finding depends on Qwen.

Only run this if resources allow.

## 20.1 Compare

```text
Semantic + Qwen
Semantic + DeepSeek
Semantic + StarCoder2

Graph-Aware + Qwen
Graph-Aware + DeepSeek
Graph-Aware + StarCoder2
```

## 20.2 Keep Retrieval Fixed

Do not change retrieval algorithms between generators.

## Deliverable

Generator robustness results.

## Exit Condition

Results can be reported as a secondary experiment rather than mixed into the main comparison.

---

# PHASE 21 — Error Analysis

## Objective

Understand **why** systems succeed or fail.

Do not stop at numerical averages.

## 21.1 Error Categories

Classify failures such as:

```text
1. Wrong file retrieved
2. Correct file but wrong function/class
3. Missing dependency
4. Missing caller/callee
5. Missing inheritance relationship
6. Multi-hop relationship required
7. Too much irrelevant context
8. Correct context but wrong answer
9. Hallucination
10. Unresolved reference issue
```

## 21.2 Compare Retrieval Methods

Look for cases where:

```text
BM25 succeeds
Semantic fails

Semantic succeeds
Graph-Aware fails

Graph-Aware succeeds
Semantic fails
```

## 21.3 Focus on Graph-Aware Cases

Especially identify questions where the answer requires:

```text
A → B
B → C
```

rather than only one directly relevant code chunk.

## Deliverable

Qualitative error-analysis table.

## Exit Condition

The final paper can explain not only **whether** graph-aware retrieval helps, but **where and why** it helps.

---

# PHASE 22 — Generate Final Results

## Objective

Convert raw experiment data into research tables and figures.

## 22.1 Retrieval Results Table

```text
| Method | Precision@K | Recall@K | MRR |
|--------|-------------|----------|-----|
| BM25 | | | |
| Semantic | | | |
| Structural | | | |
| Hybrid | | | |
| Graph-Aware | | | |
```

## 22.2 Answer Results Table

```text
| Method | Correctness | Completeness | Faithfulness |
|--------|-------------|--------------|--------------|
| BM25 | | | |
| Semantic | | | |
| Structural | | | |
| Hybrid | | | |
| Graph-Aware | | | |
```

## 22.3 Efficiency Table

Include relevant measurements such as:

```text
context size
latency
retrieval time
generation time
```

## 22.4 Ablation Table

```text
| Retrieval | Encoder | Correctness | Completeness | Faithfulness |
|-----------|---------|-------------|--------------|--------------|
| Semantic | CodeBERT | | | |
| Semantic | GraphCodeBERT | | | |
| Graph-Aware | CodeBERT | | | |
| Graph-Aware | GraphCodeBERT | | | |
```

## 22.5 Figures

Potential figures:

- retrieval metric comparison
- answer quality comparison
- context efficiency
- graph-aware vs semantic comparison
- error category distribution

## Exit Condition

All final tables and figures are generated directly from saved experiment results.

---

# PHASE 23 — Analyze the Results

## Objective

Turn numbers into a research interpretation.

## 23.1 Main Question

Does graph-aware retrieval improve repository-level QA compared with conventional retrieval?

## 23.2 Analyze Retrieval

Compare:

```text
Precision
Recall
MRR
```

## 23.3 Analyze Answer Quality

Compare:

```text
Correctness
Completeness
Faithfulness
```

## 23.4 Analyze Context Efficiency

Determine whether graph-aware retrieval improves useful context without simply increasing context size.

## 23.5 Analyze Failure Cases

Use Phase 21.

## 23.6 Avoid Unsupported Claims

Do not make universal claims. Interpret results only for the tested benchmark, repositories, models, and configurations.

## Deliverable

Results interpretation document.

---

# PHASE 24 — Reproducibility and Final Validation

## Objective

Ensure another team member can reproduce the experiment.

## 24.1 Freeze Code

Tag the final experimental implementation.

## 24.2 Freeze Configuration

Save:

```text
retrieval parameters
K
context budget
model versions
generation parameters
evaluation settings
```

## 24.3 Save Raw Results

Never overwrite raw experimental results.

## 24.4 Verify Pipeline

Run:

```text
Repository
→ Analyzer
→ JSON
→ KG
→ Retrieval
→ Context
→ LLM
→ Evaluation
```

from a clean environment where practical.

## 24.5 Check Data Integrity

Verify:

- no missing question IDs
- no duplicate results
- no accidental configuration mixing
- all failed runs documented
- model names recorded
- retrieval method recorded

## Deliverable

Reproducible final experiment package.

## Exit Condition

The full experiment can be explained and rerun from configuration files and saved code/data.

---

# PHASE 25 — Research Paper

## Objective

Convert the implementation and experiments into the final paper.

## 25.1 Introduction

Explain:

- repository-level software understanding
- limitations of purely lexical/semantic retrieval
- importance of repository relationships
- motivation for graph-aware retrieval

## 25.2 Related Work

Cover:

- code retrieval
- repository-level QA
- code embeddings
- software graphs
- Knowledge Graph-based retrieval
- graph-aware code understanding

## 25.3 Methodology

Describe:

```text
Repository Analysis
      ↓
Knowledge Graph
      ↓
Retrieval Methods
      ↓
Context Builder
      ↓
LLM
      ↓
Evaluation
```

## 25.4 Knowledge Graph

### Nodes

```text
Repository
Directory
File
Class
Function
Method
```

### Relationships

```text
CONTAINS
IMPORTS
INHERITS
CALLS
DEPENDS_ON
```

## 25.5 Retrieval Methods

Explain each:

1. BM25
2. Semantic
3. Structural
4. Hybrid
5. Graph-Aware

## 25.6 Experimental Setup

Report:

- dataset
- repositories
- question count
- models
- retrieval settings
- context budget
- evaluation metrics

## 25.7 Results

Include final tables and figures.

## 25.8 Ablation

Discuss CodeBERT vs GraphCodeBERT.

## 25.9 Error Analysis

Show representative cases.

## 25.10 Limitations

Discuss:

- benchmark limitations
- repository language/scope
- KG extraction limitations
- unresolved references
- computational constraints
- model dependence
- retrieval parameter sensitivity

## 25.11 Conclusion

Answer the research question based strictly on measured evidence.

---

# 7. Final End-to-End Execution Order

```text
PHASE 0
Freeze scope
      ↓
PHASE 1
Environment
      ↓
PHASE 2
Analyzer contract
      ↓
PHASE 3
KG implementation
      ↓
PHASE 4
KG testing
      ↓
PHASE 5
Visualization
      ↓
PHASE 6
Benchmark + repository preparation
      ↓
PHASE 7
Common retrieval interface
      ↓
PHASE 8
BM25
      ↓
PHASE 9
Semantic + CodeBERT
      ↓
PHASE 10
Structural
      ↓
PHASE 11
Hybrid
      ↓
PHASE 12
Graph-Aware
      ↓
PHASE 13
Context Builder
      ↓
PHASE 14
Qwen generation
      ↓
PHASE 15
Retrieval evaluation
      ↓
PHASE 16
Answer evaluation
      ↓
PHASE 17
Pilot
      ↓
PHASE 18
Main experiment
      ↓
PHASE 19
GraphCodeBERT ablation
      ↓
PHASE 20
Optional generator experiment
      ↓
PHASE 21
Error analysis
      ↓
PHASE 22
Results
      ↓
PHASE 23
Analysis
      ↓
PHASE 24
Reproducibility
      ↓
PHASE 25
Paper
```

---

# 8. What Is Already Complete

The following work is already substantially complete and should **not be rebuilt unnecessarily**.

## Repository Analysis

The normalized analyzer output contract is finalized.

## Knowledge Graph

Current implementation includes:

```text
builder.py
graph.py
models.py
queries.py
schema.py
serialization.py
traversal.py
validation.py
```

## Testing

Current KG baseline:

```text
39 tests
39 passed
0 failures
```

## Generated Graph

Current generated artifacts include:

```text
data/generated/repository_graph.graphml
data/generated/repository_graph.json
```

## Sample Analyzer Output

Canonical fixture:

```text
data/sample/sample_analysis.json
```

---

# 9. Definition of Done

ProjectGenome is considered implementation-complete when:

## Repository Understanding

- [ ] Repository analyzer produces normalized JSON.
- [ ] Analyzer contract is stable.
- [ ] Provenance is preserved.

## Knowledge Graph

- [ ] NetworkX MultiDiGraph is built automatically.
- [ ] Required node types are supported.
- [ ] Required relationships are supported.
- [ ] DEPENDS_ON is derived correctly.
- [ ] Unresolved references remain unresolved.
- [ ] Graph serialization works.
- [ ] Tests pass.
- [ ] Visualization works.

## Retrieval

- [x] Common retrieval interface exists.
- [x] BM25 works.
- [x] Semantic CodeBERT retrieval works.
- [x] Structural retrieval works.
- [ ] Hybrid retrieval works.
- [ ] Graph-Aware retrieval works.
- [ ] All methods return common result structures.

## Generation

- [ ] Context Builder is shared.
- [ ] Qwen pipeline works.
- [ ] Answers are saved with metadata.
- [ ] Generation configuration is recorded.

## Evaluation

- [ ] Precision@K implemented.
- [ ] Recall@K implemented.
- [ ] MRR implemented.
- [ ] Correctness evaluated.
- [ ] Completeness evaluated.
- [ ] Faithfulness evaluated.
- [ ] Context efficiency evaluated.

## Experiments

- [ ] Pilot completed.
- [ ] Main experiment completed.
- [ ] GraphCodeBERT ablation completed.
- [ ] Optional generator experiment completed if resources allow.
- [ ] Error analysis completed.

## Research

- [ ] Final tables generated.
- [ ] Final figures generated.
- [ ] Results interpreted.
- [ ] Limitations documented.
- [ ] Paper completed.

---

# 10. Important Rules for the Team

## Rule 1 — Do Not Change the Main Experiment Casually

The main comparison is:

```text
BM25
vs
Semantic
vs
Structural
vs
Hybrid
vs
Graph-Aware
```

with CodeBERT and Qwen.

## Rule 2 — Do Not Mix Retrieval and Generation Effects

The main experiment keeps the generator fixed.

The research question is primarily about retrieval.

## Rule 3 — Do Not Treat the KG as the Final Answer

The KG is an intermediate representation used to retrieve and expand repository context.

## Rule 4 — Graph-Aware Retrieval Must Actually Use the Graph

It should follow:

```text
semantic seed
    ↓
KG mapping
    ↓
relationship traversal
    ↓
context expansion
```

Simply searching graph node names is not sufficient.

## Rule 5 — Keep Provenance

Every retrieved item should be traceable to:

```text
repository
→ file
→ entity
→ source location
```

Graph-expanded evidence should additionally preserve the relationship path.

## Rule 6 — Do Not Invent Results

No accuracy, recall, faithfulness, or improvement number should appear until the corresponding experiment has actually been run.

## Rule 7 — Pilot Before Full Experiment

Do not immediately run thousands of generations.

First run the pilot.

## Rule 8 — Do Not Over-Engineer

The goal is a valid research experiment, not an unnecessarily large production system.

---

# 11. Final Research Story

The final paper should tell one clear story:

```text
Traditional retrieval
        ↓
finds relevant code
        ↓
but repository understanding also requires relationships
        ↓
Knowledge Graph represents those relationships
        ↓
Graph-Aware Retrieval expands semantic results through those relationships
        ↓
controlled experiment compares it with conventional retrieval
        ↓
retrieval + answer metrics determine whether the approach helps
```

The core contribution is therefore the **controlled investigation of whether graph-aware repository context improves repository-level software question answering**, rather than simply claiming that building a Knowledge Graph is novel.

---

# 12. Immediate Next Step

Since the Knowledge Graph and its tests are already complete, the next practical phase is:

```text
PHASE 6
Benchmark + Repository Preparation
```

Proceed incrementally:

```text
1. Verify SWE-QA structure
2. Map questions → repositories
3. Prepare repository retrieval units
4. Establish ground-truth evidence
5. Build common retrieval interface
6. Implement BM25
7. Test BM25 end-to-end
8. Implement CodeBERT
9. Test Semantic retrieval
10. Continue to Structural → Hybrid → Graph-Aware
```

Do not implement all five retrieval methods simultaneously. Each method should be verified before the next one is started.
