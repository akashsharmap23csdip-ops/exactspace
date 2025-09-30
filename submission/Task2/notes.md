# RAG System Design Notes

## Document Processing Strategy

### Chunking Approach
- **Size**: Hybrid chunking strategy with 500-700 tokens per chunk
- **Overlap**: 20% overlap between consecutive chunks to maintain context
- **Granularity**: Semantic chunking that respects document structure (paragraphs, sections)
- **Rationale**: Balances retrieval precision with maintaining sufficient context; small enough for targeted retrieval but large enough to preserve meaning

### Embedding Model
- **Selected Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Rationale**: Good balance between performance and efficiency; performs well on technical documentation while being lightweight enough for local deployment

### Retrieval Method
- **Primary**: Dense vector search using cosine similarity
- **Secondary**: Hybrid retrieval combining vector search with BM25 lexical search
- **Reranking**: Cross-encoder reranking on top-k results to improve relevance
- **Rationale**: Vector search captures semantic similarity while BM25 ensures coverage of keyword matches; reranking helps prioritize the most relevant chunks

## Ensuring Relevance and Faithfulness

1. **Metadata Enhancement**
   - Add document title, section headers, and document type to each chunk
   - Include page numbers and section context for accurate citations

2. **Source Tracking**
   - Maintain clear provenance of each chunk (document name, page number)
   - Return source snippets alongside generated answers

3. **Grounding Techniques**
   - Enforced citation formatting in LLM prompt
   - Answer verification step comparing response to retrieved contexts
   - Confidence scoring for each part of the response

## Guardrails & Failure Modes

### No Relevant Answers
- Implement a relevance threshold for retrieved documents
- Fallback message: "I don't have enough information to answer this question confidently. Here's what I can tell you about related topics..."
- Suggest alternative questions based on available information

### Hallucinations
- Enforce citation for every factual statement
- Implement fact verification by comparing response to source documents
- Clearly separate retrieved information from LLM-generated inferences

### Sensitive Queries
- Define blocklist for inappropriate or dangerous queries
- Implement content classifiers to detect sensitive topics
- Provide templated responses for out-of-scope questions

### Monitoring Metrics
- **Precision@k**: Accuracy of top k retrieved documents
- **Recall@k**: Coverage of relevant documents in top k results
- **Response faithfulness**: Percentage of statements with valid citations
- **User feedback**: Track satisfaction and correction rates

## Scalability Considerations

### 10x Document Increase
- Implement document clustering for semantic partitioning of the corpus
- Use approximate nearest neighbor search (HNSW algorithm)
- Implement hierarchical indexing to maintain low latency
- Consider distributed vector storage

### 100+ Concurrent Users
- Implement connection pooling for vector database
- Cache frequent queries and their results
- Pre-compute embeddings for all documents
- Horizontal scaling of the retrieval service

### Cost-Efficient Cloud Deployment
- Use serverless architecture for bursty workloads
- Optimize GPU usage with batching for embedding generation
- Implement tiered storage (hot/warm/cold) based on document access patterns
- Consider quantization of embeddings to reduce storage requirements

## Architecture Diagram

The architecture diagram is provided in `architecture_diagram.pptx`, showing:
1. Document ingestion pipeline
2. Vector database and indexing
3. Query processing flow
4. LLM integration
5. Monitoring and feedback systems