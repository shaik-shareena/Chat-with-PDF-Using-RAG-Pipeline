# **Chat-with-PDF-Using-RAG-Pipeline**
#**Overview**

The goal is to implement a Retrieval-Augmented Generation (RAG) pipeline that allows users to
interact with semi-structured data in multiple PDF files. The system should extract, chunk,
embed, and store the data for eFicient retrieval. It will answer user queries and perform
comparisons accurately, leveraging the selected LLM model for generating responses.

**Functional Requirements**
**1. Data Ingestion**
**• Input:**PDF files containing semi-structured data.
**Process:**
o Extract text and relevant structured information from PDF files.
