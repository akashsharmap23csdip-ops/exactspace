# RAG System Prototype

This folder contains a prototype implementation of a Retrieval-Augmented Generation (RAG) system for querying technical documents with natural language.

## Overview

The prototype demonstrates:
- Document loading and processing
- Text chunking with overlaps
- Vector embeddings and storage
- Retrieval of relevant context
- LLM-based answer generation with citations

## Architecture

The system uses a modular design with the following components:

1. **DocumentLoader**: Loads PDF files and extracts text with metadata
2. **DocumentChunker**: Splits documents into semantically meaningful chunks
3. **VectorStore**: Embeds and indexes chunks for efficient retrieval
4. **RAGSystem**: Combines retrieval and generation to answer queries

## Installation

### Prerequisites

- Python 3.8+
- Required Python packages:

```bash
pip install transformers sentence-transformers chromadb pymupdf langchain pandas numpy
```

### Models

The system uses:
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (automatically downloaded)
- **LLM**: `google/flan-t5-small` (automatically downloaded)

## Running the Prototype

1. Place your PDF technical documents in the `/workspaces/exactspace/Task2/docs` directory
2. Run the prototype:

```bash
cd /workspaces/exactspace
python Task2/prototype/rag_prototype.py
```

3. Enter your questions when prompted

## Sample Documents

If no PDF documents are available, the system creates a sample text document for demonstration purposes. For a real application, you should provide actual PDF technical documents.

## Evaluation

The prototype includes a simple evaluation mechanism that measures:
- Average retrieval time
- Average generation time
- Total processing time

This can be extended with more sophisticated metrics like precision@k and recall@k in a production implementation.

## Limitations and Future Improvements

1. **Document Loading**: Currently limited to PDFs; could be extended to more formats
2. **Chunking Strategy**: Using basic character-based chunking; could implement semantic chunking
3. **Retrieval**: Simple vector similarity; could implement hybrid retrieval with BM25
4. **LLM**: Using a small model for demonstration; larger models would improve answer quality
5. **Evaluation**: Simple performance metrics; could add relevance and faithfulness metrics

## Directory Structure

- `rag_prototype.py`: Main implementation of the RAG system
- `../docs/`: Directory for technical documents (PDFs)