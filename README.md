````markdown
# ExactSpace Data Science Assessment

This repository contains the implementation of a two-part assessment:

1. **Machine Data Analysis**: Analysis of cyclone machine sensor data for state detection, anomaly detection, and forecasting
2. **RAG + LLM System Design**: Design and prototype of a Retrieval-Augmented Generation system for technical documentation

## Implementation Summary

### Task 1: Machine Data Analysis
- ✅ Loaded and processed 3 years of cyclone sensor data (~370,000 records)
- ✅ Detected 207 shutdown periods with detailed statistics
- ✅ Identified 4 distinct operational states using KMeans clustering
- ✅ Performed anomaly detection using Isolation Forest
- ✅ Developed forecasting models (Persistence, ARIMA, Prophet)
- ✅ Generated visualizations and actionable insights

### Task 2: RAG System Design
- ✅ Designed comprehensive architecture for document retrieval and generation
- ✅ Implemented document loading, chunking, and vector embedding
- ✅ Created prototype using sentence-transformers and FLAN-T5
- ✅ Developed evaluation metrics for retrieval quality
- ✅ Provided detailed scaling and deployment considerations

## Repository Structure

```
├── data.xlsx                      # Cyclone sensor data (3 years at 5-min intervals)
├── Task1/                         # Machine Data Analysis
│   ├── task1_analysis.py          # Main analysis script
│   ├── README.md                  # Task 1 documentation
│   └── plots/                     # Visualization outputs
├── Task2/                         # RAG + LLM System Design
│   ├── architecture_diagram.html  # Architecture diagram (HTML version)
│   ├── notes.md                   # Design notes and trade-offs
│   ├── evaluation.csv             # Retrieval evaluation metrics
│   ├── docs/                      # Sample technical documents
│   └── prototype/                 # RAG system implementation
│       ├── rag_prototype.py       # Main prototype code
│       └── README.md              # Prototype documentation
```

## Task 1: Machine Data Analysis

This task analyzes 3 years of cyclone machine sensor data (approximately 370,000 records at 5-minute intervals) to:

1. **Data Preparation & Exploratory Analysis**
   - Load, clean, and explore the dataset
   - Handle missing values and outliers
   - Generate summary statistics and visualizations

2. **Shutdown/Idle Period Detection**
   - Detect and segment shutdown periods
   - Calculate total downtime and frequency statistics

3. **Machine State Segmentation (Clustering)**
   - Apply clustering to identify operational states
   - Characterize each state with summary statistics

4. **Contextual Anomaly Detection**
   - Build context-aware anomaly detection
   - Perform root cause analysis on selected anomalies

5. **Short-Horizon Forecasting**
   - Forecast Cyclone_Inlet_Gas_Temp one hour ahead
   - Compare multiple forecasting models

6. **Insights & Recommendations**
   - Provide actionable insights based on analysis
   - Suggest monitoring improvements

### Running Task 1

```bash
cd /workspaces/exactspace
python Task1/task1_analysis.py
```

See the Task1/README.md file for detailed instructions.

## Task 2: RAG + LLM System Design

This task designs and implements a prototype Retrieval-Augmented Generation (RAG) system for querying technical documentation:

1. **System Architecture**
   - Complete architecture design for document processing, embedding, retrieval, and LLM integration
   - Chunking strategy and embedding model selection

2. **Retrieval Strategy**
   - Hybrid vector and lexical search
   - Reranking for improved relevance

3. **Guardrails & Failure Modes**
   - Handling of no relevant answers
   - Prevention of hallucinations
   - Sensitive query filtering

4. **Scalability Considerations**
   - Handling increased document volume
   - Supporting concurrent users
   - Cloud deployment strategies

5. **Prototype Implementation**
   - Document loading and processing
   - Vector indexing and retrieval
   - LLM-based answer generation with citations

### Running Task 2 Prototype

```bash
cd /workspaces/exactspace
python Task2/prototype/rag_prototype.py
```

See the Task2/prototype/README.md file for detailed instructions.

## Requirements

### Task 1
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- statsmodels
- prophet

### Task 2
- transformers
- sentence-transformers
- chromadb
- pymupdf
- langchain

Install all dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy statsmodels prophet transformers sentence-transformers chromadb pymupdf langchain
```