"""
RAG Prototype Implementation

This script implements a minimal RAG (Retrieval-Augmented Generation) system
that can ingest technical documents, index them in a vector database,
and answer natural language queries with citations to the source material.

The implementation uses:
- Sentence Transformers for embeddings
- Chroma as the vector database
- FLAN-T5-Small as the LLM for answering questions
"""

import os
import sys
import re
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

# Document loading and processing
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Embeddings and vector database
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# LLM
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Create necessary directories
os.makedirs("../docs", exist_ok=True)


class DocumentLoader:
    """Loads and processes documents for RAG system."""
    
    def __init__(self, docs_dir: str = "../docs"):
        self.docs_dir = docs_dir
    
    def load_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Load a PDF file and extract text with page numbers."""
        print(f"Loading PDF: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # Skip empty pages
                pages.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "source": os.path.basename(pdf_path)
                })
        
        print(f"Extracted {len(pages)} pages from {pdf_path}")
        return pages
    
    def load_all_pdfs(self) -> List[Dict[str, Any]]:
        """Load all PDFs in the docs directory."""
        all_pages = []
        for file in os.listdir(self.docs_dir):
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(self.docs_dir, file)
                pages = self.load_pdf(pdf_path)
                all_pages.extend(pages)
        
        print(f"Total pages loaded: {len(all_pages)}")
        return all_pages


class DocumentChunker:
    """Splits documents into chunks suitable for embedding."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    
    def create_chunks(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split pages into chunks with metadata."""
        all_chunks = []
        
        for page in pages:
            chunks = self.text_splitter.split_text(page["text"])
            
            for i, chunk_text in enumerate(chunks):
                chunk = {
                    "text": chunk_text,
                    "source": page["source"],
                    "page_num": page["page_num"],
                    "chunk_id": f"{page['source']}_p{page['page_num']}_c{i+1}"
                }
                all_chunks.append(chunk)
        
        print(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks


class VectorStore:
    """Manages the vector database for document retrieval."""
    
    def __init__(self, collection_name: str = "technical_docs"):
        # Initialize embedding function using sentence-transformers
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize Chroma client
        self.client = chromadb.Client()
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {collection_name}")
        except ValueError:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: {collection_name}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [
            {
                "source": chunk["source"],
                "page_num": str(chunk["page_num"])
            } for chunk in chunks
        ]
        
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end_idx],
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def query(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        """Retrieve relevant documents for a query."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        
        return results


class RAGSystem:
    """Main RAG system that combines document loading, chunking, and retrieval with LLM generation."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        self.document_loader = DocumentLoader()
        self.document_chunker = DocumentChunker()
        self.vector_store = VectorStore()
        
        # Load LLM
        print(f"Loading LLM: {model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        print("LLM loaded successfully")
    
    def ingest_documents(self) -> None:
        """Ingest all PDFs from the docs directory."""
        # Load documents
        pages = self.document_loader.load_all_pdfs()
        
        # Create chunks
        chunks = self.document_chunker.create_chunks(pages)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        
        print(f"Ingested {len(pages)} pages, created {len(chunks)} chunks")
    
    def generate_answer(self, query: str, n_results: int = 5) -> str:
        """Generate an answer to the query using RAG."""
        # Retrieve relevant chunks
        results = self.vector_store.query(query, n_results=n_results)
        
        # Extract documents and their metadata
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        # Build prompt for T5
        context = "\n\n".join([f"Document: {meta['source']}, Page: {meta['page_num']}\n{doc}" 
                             for doc, meta in zip(documents, metadatas)])
        
        prompt = f"Answer the following question based on the provided context. Include citations to source documents in your answer.\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        # Tokenize and generate answer
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).input_ids
        
        output = self.model.generate(
            input_ids,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
        
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Add source citations if not present in the answer
        if not any(f"Document: {meta['source']}" in answer for meta in metadatas):
            answer += "\n\nSources:"
            for i, meta in enumerate(metadatas):
                answer += f"\n[{i+1}] {meta['source']}, Page {meta['page_num']}"
        
        return answer
    
    def evaluate(self, test_queries: List[str]) -> Dict[str, float]:
        """Evaluate the RAG system on a set of test queries."""
        # This is a simplified evaluation that could be expanded
        results = {
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
            "total_time": 0
        }
        
        total_retrieval_time = 0
        total_generation_time = 0
        total_time_start = time.time()
        
        for query in test_queries:
            # Measure retrieval time
            retrieval_start = time.time()
            retrieved_docs = self.vector_store.query(query, n_results=5)
            retrieval_time = time.time() - retrieval_start
            total_retrieval_time += retrieval_time
            
            # Measure generation time
            generation_start = time.time()
            _ = self.generate_answer(query)
            generation_time = time.time() - generation_start
            total_generation_time += generation_time
        
        total_time = time.time() - total_time_start
        num_queries = len(test_queries)
        
        results["avg_retrieval_time"] = total_retrieval_time / num_queries
        results["avg_generation_time"] = total_generation_time / num_queries
        results["total_time"] = total_time
        
        return results


def create_sample_docs():
    """Create sample technical documents for testing if none exist."""
    docs_dir = Path("../docs")
    docs_dir.mkdir(exist_ok=True)
    
    if not any(docs_dir.glob("*.pdf")):
        print("No PDF documents found. Creating sample documents...")
        
        # Create a simple text file as a placeholder
        with open(docs_dir / "sample_technical_doc.txt", "w") as f:
            f.write("""
            # Cyclone Operation Manual
            
            ## Introduction
            
            A cyclone is a device that uses centrifugal force to separate particles from a gas stream. It consists of an inlet, a cylindrical section, a conical section, and an outlet.
            
            ## Operating Parameters
            
            ### Temperature
            
            The cyclone inlet gas temperature should be maintained between 300°C and 450°C for optimal operation. If the temperature drops below 300°C, incomplete combustion may occur, leading to increased emissions. If the temperature exceeds 450°C, thermal damage to the cyclone lining may occur.
            
            ### Pressure
            
            The cyclone inlet draft should be maintained between -150 mmWG and -130 mmWG. A sudden drop in draft may indicate a blockage or a leak in the system. Conversely, a sudden increase in draft (becoming less negative) might indicate a draft control issue.
            
            ### Material Flow
            
            The material temperature should be monitored closely. A high material temperature (>380°C) might indicate excessive heat transfer from the gas, which could affect downstream equipment.
            
            ## Troubleshooting
            
            ### Scenario 1: Sudden Draft Drop
            
            If there is a sudden drop in cyclone inlet draft (becoming more negative), check for:
            1. Partial blockage in downstream equipment
            2. Improper fan operation
            3. Changes in upstream process conditions
            
            ### Scenario 2: High Outlet Temperature
            
            If the cyclone gas outlet temperature is abnormally high:
            1. Check for reduced material flow
            2. Verify the inlet gas temperature
            3. Inspect refractory lining for damage
            
            ## Maintenance
            
            Regular maintenance should be performed weekly to ensure optimal operation. This includes:
            1. Inspection of the cyclone body for wear
            2. Checking the draft gauge for proper functioning
            3. Calibration of temperature sensors
            """)
        
        print("Created a sample document in text format")
        print("Note: For a real application, you would need actual PDF documents")


def main():
    # Check for sample documents
    create_sample_docs()
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    # Ingest documents
    rag_system.ingest_documents()
    
    # Interactive query answering
    print("\nRAG System Ready for Queries! (Type 'exit' to quit)")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        # Generate and print answer
        print("\nGenerating answer...\n")
        answer = rag_system.generate_answer(query)
        print(answer)
    
    # Simple evaluation
    test_queries = [
        "What is the optimal temperature range for cyclone operation?",
        "What should I check if there's a sudden drop in draft?",
        "How often should maintenance be performed?",
    ]
    
    print("\nRunning evaluation on test queries...")
    eval_results = rag_system.evaluate(test_queries)
    
    print(f"\nEvaluation Results:")
    print(f"Average Retrieval Time: {eval_results['avg_retrieval_time']:.4f} seconds")
    print(f"Average Generation Time: {eval_results['avg_generation_time']:.4f} seconds")
    print(f"Total Processing Time: {eval_results['total_time']:.4f} seconds")


if __name__ == "__main__":
    main()