import os
import streamlit as st
import pandas as pd
import io
import time
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
import concurrent.futures
import traceback

# Initialize the Qdrant client and SentenceTransformer model
@st.cache_resource
def initialize_qdrant_and_encoder():
    encoder = SentenceTransformer("paraphrase-MiniLM-L6-V2")
    qdrant = QdrantClient(":memory:")
    qdrant.recreate_collection(
        collection_name="industries",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    return qdrant, encoder

# Batch encoding with progress display
def encode_in_batches(text_data, encoder, batch_size=128):
    vectors = []
    for i in range(0, len(text_data), batch_size):
        batch = text_data[i:i + batch_size]
        batch_vectors = encoder.encode(batch).tolist()
        vectors.extend(batch_vectors)
        # Show progress for encoding
        show_progress_bar("Encoding data", i + batch_size, len(text_data))
    return vectors

# Upsert in batches to Qdrant
def upsert_in_batches(vectors, qdrant, data_dict, batch_size=128):
    for i in range(0, len(vectors), batch_size):
        batch = [
            models.PointStruct(
                id=j,
                vector=vectors[j],
                payload={
                    "isic_code": data_dict[j]['isic_code'],
                    "title": data_dict[j]['title']
                }
            ) for j in range(i, min(i + batch_size, len(vectors)))
        ]
        qdrant.upsert(collection_name="industries", points=batch)
        # Show progress for upserting
        show_progress_bar("Upserting vectors to Qdrant", i + batch_size, len(vectors))

# Progress bar function
def show_progress_bar(message, current, total):
    progress_value = min(current / total, 1.0)
    st.progress(progress_value)
    st.write(f"{message}: {int(progress_value * 100)}% complete.")

# Load and encode ISIC data
@st.cache_resource
def load_and_encode_isic_data(isic_data_path, _qdrant, _encoder):
    start_time = time.time()

    data_df = pd.read_excel(isic_data_path, sheet_name='Sheet1')
    st.write("ISIC data loaded successfully.")

    # Clean the data
    data_df = data_df[data_df["Level"] == 2]
    data_df["text"] = data_df.apply(lambda row: " ".join([str(row["Definition"]), str(row["ExplanatoryNoteExclusion"])]), axis=1)
    data_dict = data_df[["ISIC Code", "Title EN", "text"]].rename(columns={"Title EN": "title", "ISIC Code": "isic_code"}).to_dict(orient="records")

    # Batch encode the data
    texts = [record['text'] for record in data_dict]
    vectors = encode_in_batches(texts, encoder)

    # Upsert vectors to Qdrant in batches
    upsert_in_batches(vectors, qdrant, data_dict)

    end_time = time.time()
    st.write(f"ISIC data encoded and upserted to Qdrant. Time taken: {end_time - start_time:.2f} seconds.")
    return data_df

# Find top 3 ISIC codes
def find_top_3_isic_codes(industry, encoder, qdrant):
    vector = encoder.encode(industry).tolist()
    hits = qdrant.search(collection_name="industries", query_vector=vector, limit=3)
    results = [hit.payload.get('isic_code') for hit in hits]
    return results + [None] * (3 - len(results))

# Process individual file
def process_file(uploaded_file, encoder, qdrant, save_directory, progress_container):
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)

        st.write(f"File '{uploaded_file.name}' uploaded successfully. Processing...")

        start_time = time.time()

        if 'INDUSTRY' in df.columns:
            df['combined_text'] = df['INDUSTRY'].astype(str)
            df[['isic_code_1', 'isic_code_2', 'isic_code_3']] = df['combined_text'].apply(lambda x: find_top_3_isic_codes(x, encoder, qdrant)).apply(pd.Series)

            end_time = time.time()
            st.write(f"Processing of '{uploaded_file.name}' completed in {end_time - start_time:.2f} seconds.")

            save_path = os.path.join(save_directory, uploaded_file.name)
            df.to_csv(save_path, index=False)
            st.write(f"File saved at {save_path}")
        else:
            st.error(f"The uploaded file '{uploaded_file.name}' must contain an 'INDUSTRY' column.")
    except Exception as e:
        st.error(f"Error processing file '{uploaded_file.name}': {e}")
        st.text(traceback.format_exc())

# Main application logic
st.title("Industry to ISIC Code Mapper")

# Step 1: User input for name
user_name = st.text_input("Please enter your name:")

if user_name:
    outputs_dir = "outputs"
    user_dir = os.path.join(outputs_dir, user_name)

    # Step 2: Check if user directory exists, if not, create it
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)
        st.write(f"Folder created for {user_name}.")
    else:
        st.write(f"Welcome back, {user_name}. Using existing folder.")

    # Initialize Qdrant and encoder
    qdrant, encoder = initialize_qdrant_and_encoder()

    # Load and encode ISIC data
    isic_data_path = "isic_index.xlsx"
    try:
        load_and_encode_isic_data(isic_data_path, qdrant, encoder)
    except Exception as e:
        st.error(f"Error loading and encoding ISIC data: {e}")
        st.text(traceback.format_exc())

    # File uploader for multiple files
    uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=['xlsx', 'csv'], accept_multiple_files=True)

    if uploaded_files:
        # Create a container to show progress bars for each file
        progress_container = st.container()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {executor.submit(process_file, file, encoder, qdrant, user_dir, progress_container): file for file in uploaded_files}

            # Show progress for each file
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    future.result()
                except Exception as e:
                    st.error(f"Error processing file '{file.name}': {e}")
                    st.text(traceback.format_exc())

        # Log file to track the number of files processed
        log_file_path = os.path.join(outputs_dir, "processing_log.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"{user_name} processed {len(uploaded_files)} files.\n")
