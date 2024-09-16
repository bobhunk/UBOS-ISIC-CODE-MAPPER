import streamlit as st
import pandas as pd
import traceback
import io
import torch
import numpy as np
import time  
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize the Qdrant client and SentenceTransformer model
@st.cache_resource
def initialize_qdrant_and_encoder():
    encoder = SentenceTransformer("all-mpnet-base-v2")
    qdrant = QdrantClient(":memory:")  
    qdrant.recreate_collection(
        collection_name="industries",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    return qdrant, encoder

# Batch encoding to improve speed
def encode_in_batches(text_data, encoder, batch_size=128):
    vectors = []
    total_batches = len(text_data) // batch_size + (1 if len(text_data) % batch_size != 0 else 0) 
    progress_bar = st.progress(0)  # Initialize a progress bar

    for i in range(0, len(text_data), batch_size):
        batch = text_data[i:i + batch_size]
        batch_vectors = encoder.encode(batch).tolist()
        vectors.extend(batch_vectors)
        
        # Calculate progress percentage and update the progress bar
        progress_value = min(int((i + batch_size) / len(text_data) * 100), 100)
        progress_bar.progress(progress_value)  # Ensure the progress value stays between 0 and 100
    
    return vectors

# Batch upsert into Qdrant
def upsert_in_batches(vectors, qdrant, data_dict, batch_size=64):
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

# Load and encode ISIC data
def load_and_encode_isic_data(isic_data_path, qdrant, encoder):
    # Load the ISIC data from the file
    data_df = pd.read_excel(isic_data_path, sheet_name='Sheet1')
    st.write("ISIC data loaded successfully.")
    
    # Clean the data
    data_df = data_df[data_df["Level"] == 2]
    data_df["text"] = data_df.apply(lambda row: " ".join([str(row["Definition"]), str(row["ExplanatoryNoteExclusion"])]), axis=1)
    data_df["words"] = data_df.apply(lambda row: len(row["text"].split(" ")), axis=1)
    data_dict = data_df[["ISIC Code", "Title EN", "text"]].rename(columns={"Title EN": "title", "ISIC Code": "isic_code"}).to_dict(orient="records")
    
    # Batch encode the cleaned data
    texts = [record['text'] for record in data_dict]
    vectors = encode_in_batches(texts, encoder)
    
    # Batch upsert vectors to Qdrant
    upsert_in_batches(vectors, qdrant, data_dict)
    
    st.write("ISIC data encoded and upserted to Qdrant.")
    return data_df

# Top 3 ISIC codes for a given industry
def find_top_3_isic_codes(industry, encoder, qdrant):
    vector = encoder.encode(industry).tolist()
    hits = qdrant.search(collection_name="industries", query_vector=vector, limit=3)
    results = [hit.payload.get('isic_code') for hit in hits]
    while len(results) < 3:
        results.append(None)
    return results

st.title("Industry to ISIC Code Mapper")

# Initialize Qdrant and encoder
qdrant, encoder = initialize_qdrant_and_encoder()

# Load and clean ISIC data
isic_data_path = "isic_index.xlsx"
try:
    # Start time for performance measurement
    start_time = time.time()
    
    load_and_encode_isic_data(isic_data_path, qdrant, encoder)
    
    # End time and elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.success(f"ISIC data loading and encoding completed in {elapsed_time:.2f} seconds.")
    
except Exception as e:
    st.error(f"Error loading and encoding ISIC data: {e}")
    st.text("Traceback:")
    st.text(traceback.format_exc())

# File uploader for multiple files
uploaded_files = st.file_uploader("Upload your Excel or CSV files", type=['xlsx', 'csv'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Load the file (Excel or CSV)
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            
            st.write(f"File '{uploaded_file.name}' Uploaded Successfully. Processing...")

            # Start time for processing the file
            start_time = time.time()

            # Remove empty rows and convert the 'INDUSTRY' column to string
            df = df.dropna(subset=['INDUSTRY'])
            df['INDUSTRY'] = df['INDUSTRY'].astype(str)

            # Batch processing of industry descriptions for ISIC code mapping
            industries = df['INDUSTRY'].tolist()
            isic_codes = []

            for industry in industries:
                codes = find_top_3_isic_codes(industry, encoder, qdrant)
                isic_codes.append(codes)

            # Add the ISIC codes to the dataframe
            df[['isic_code_1', 'isic_code_2', 'isic_code_3']] = pd.DataFrame(isic_codes, index=df.index)

            # End time for processing
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.success(f"File '{uploaded_file.name}' processed in {elapsed_time:.2f} seconds.")
            
            st.write("Processing Complete. Download the file below.")
            st.dataframe(df)

            # Download option (Excel or CSV)
            download_format = st.radio(f"Select download format for '{uploaded_file.name}'", ("Excel", "CSV"))

            if download_format == "Excel":
                def convert_df_to_excel(df):
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False)
                    processed_data = output.getvalue()
                    return processed_data

                excel_file = convert_df_to_excel(df)
                
                st.download_button(
                    label=f"Download Processed File '{uploaded_file.name}' as Excel",
                    data=excel_file,
                    file_name=uploaded_file.name.replace('.csv', '.xlsx').replace('.xlsx', '_processed.xlsx'),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif download_format == "CSV":
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode('utf-8')

                csv_file = convert_df_to_csv(df)
                
                st.download_button(
                    label=f"Download Processed File '{uploaded_file.name}' as CSV",
                    data=csv_file,
                    file_name=uploaded_file.name.replace('.xlsx', '.csv').replace('.csv', '_processed.csv'),
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file '{uploaded_file.name}': {e}")
            st.text("Traceback:")
            st.text(traceback.format_exc())

# Check if NumPy works correctly
try:
    np_version = np.__version__
    st.success(f"NumPy version {np_version} is working!")
except Exception as e:
    st.error(f"Error importing NumPy: {e}")