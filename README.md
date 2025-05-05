# Basic Document Similarity
This project enables users to ask simple questions about cricket players and find the most relevant document using embeddings and cosine similarity.

# 1. Project Description
This script compares a user's query with predefined cricket-related documents. It uses sentence embeddings and cosine similarity to return the most relevant document based on the query.

# 2. Installation
Install the required libraries:

pip install langchain langchain-huggingface sentence-transformers scikit-learn python-dotenv
# 3. How It Works
Load cricket player-related documents.

-Convert documents into embeddings using the sentence-transformers/all-MiniLM-L6-v2 model.

-Accept a user query and embed it.

-Calculate cosine similarity between the query embedding and document embeddings.

-Return the document with the highest similarity score as the most relevant answer.

# 4. Usage Instructions

document_similarity.py
-Enter a basic cricket-related question when prompted. Example questions:

-Who is the best bowler from India?

-Which player is a great all-rounder from Australia?

-The most relevant document will be printed based on your question.

# 5. Example Output

-ask basic question about cricket about players: who is the best spinner from Australia?
-using full_answer method:  Shane Warne is best spinner so far in Australia
-using index-answer method:  Shane Warne is best spinner so far in Australia
# 6. Notes
A .env file is optional if future versions require Hugging Face API keys.

The script uses a locally available model and does not require API calls.
