UBOS ISIC Code Mapper

This project provides a web interface for mapping industry descriptions to ISIC (International Standard Industrial Classification) codes using Qdrant and SentenceTransformers. It allows users to upload files containing industry descriptions and automatically maps them to relevant ISIC codes using semantic search techniques.

Features

Batch Processing: Efficiently encodes industry descriptions and upserts data into the Qdrant vector database using SentenceTransformers.
File Upload: Supports uploading files in Excel (.xlsx) and CSV formats, which contain industry descriptions for processing.
ISIC Code Mapping: The application maps industries and descriptions to the top 3 most relevant ISIC codes using cosine similarity.
Download Processed Data: After processing, users can download the mapped ISIC codes as an Excel or CSV file.
Time Tracking: Displays the time taken to load ISIC data and process files.

Technologies Used

Streamlit - For building the web interface.
Qdrant - A vector search engine to store and search vector embeddings.
SentenceTransformers - To encode industry descriptions into vector embeddings.
Pandas - For data manipulation and processing.
OpenAI - For semantic search and vector comparisons (optional, if integrated for fine-tuning).

How It Works

ISIC Data Load and Encode: The app loads the ISIC structure and definitions from an external source. It uses SentenceTransformers to encode the definitions into vector embeddings, which are then stored in Qdrant for similarity search.

File Upload: Users can upload their own Excel or CSV files containing two columns: INDUSTRY and DESCRIPTION. These fields are concatenated and then encoded.

ISIC Code Search: Each industry description is encoded and compared against the pre-stored ISIC definitions using cosine similarity. The app returns the top 3 ISIC codes based on the similarity of vectors.

Download Processed Data: Once the industry descriptions are processed, users can download the file with the matched ISIC codes.
