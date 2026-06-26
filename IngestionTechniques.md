Good question to step back on. Here are the main alternatives to raw chunk embeddings:

---

**1. Contextual chunk embeddings**

Instead of embedding the chunk alone, prepend context before embedding:

```
"This passage is from Mahabharata, Kurukshetra War section, 
Chapter 12. Arjuna is speaking to Krishna about his duty.
[raw chunk text]"
```

Same chunk, richer embedding. Anthropic published this as **Contextual Retrieval**. Significant retrieval improvement for almost zero extra work.

---

**2. Late chunking**

Embed the *entire document* first, then chunk the embeddings — not the text. Each chunk's embedding carries full-document context because it was computed before chunking happened. `jina-ai` pioneered this. Works well for long narratives where meaning depends on what came before.

---

**3. Summary embeddings**

Instead of embedding raw text, embed an LLM/extractive summary of the chunk. The embedding represents *meaning* rather than *words*. Better for thematic/conceptual queries. You retrieve by meaning, then return the original raw text to the LLM.

---

**4. Hypothetical Document Embeddings (HyDE)**

At query time, generate a hypothetical answer, embed that instead of the question. Closes the gap between question-phrasing and answer-phrasing. Doesn't change ingestion at all.

---

**5. Multi-vector / ColBERT style**

Instead of one embedding per chunk, generate one embedding *per token* in the chunk. At query time, score using MaxSim — each query token finds its best matching chunk token. Much more precise than single-vector similarity. **ColBERT / RAGatouille** is the main implementation.

```
Single vector:  [chunk] → 1 embedding
ColBERT:        [chunk] → N embeddings (one per token)
```

Expensive to store but retrieval quality is substantially better, especially for precise factual questions.

---

**6. Sparse learned embeddings (SPLADE)**

Like BM25 but learned. Produces sparse vectors where each dimension corresponds to a vocabulary term, but weights are learned rather than TF-IDF. Combines the interpretability of BM25 with the semantic awareness of dense embeddings. Often outperforms both individually.

---

**7. Parent-child chunk embeddings**

Embed small chunks (precise retrieval) but at query time return the parent chunk (more context):

```
Ingest:   embed sentence-level chunks
Retrieve: find matching sentences
Return:   the paragraph/section they belong to
```

Solves the tension between retrieval precision (small chunks) and answer quality (large context).

---

**8. Multi-granularity embeddings**

Embed the same content at multiple levels simultaneously:

```
sentence  → embedded
paragraph → embedded  
chapter   → embedded
```

Query hits all levels, results merged. Different questions naturally resolve at different granularities. "What did Rama say to Sita" → sentence level. "What is the theme of Ayodhya Kanda" → chapter level.

---

**How they compare**

| Technique | Retrieval Quality | Storage Cost | Ingestion Cost | Query Cost |
|---|---|---|---|---|
| Raw chunks | baseline | baseline | low | low |
| Contextual chunks | ++ | same | medium | low |
| Late chunking | + | same | low | low |
| Summary embeddings | + (thematic) | same | medium | low |
| HyDE | + | same | none | medium |
| ColBERT | +++ | high | medium | medium |
| SPLADE | ++ | medium | low | low |
| Parent-child | ++ | low | low | low |
| Multi-granularity | +++ | high | low | low |

---

**For your learning project**

Start here in order:

1. **Contextual chunks** — highest ROI, minimal change to your pipeline
2. **Parent-child** — solves your context assembly problem directly
3. **Multi-granularity** — natural fit for books with chapter structure
4. **ColBERT** — when you want to understand the precision ceiling

These compound. Contextual + parent-child + multi-granularity together is what serious production systems run.