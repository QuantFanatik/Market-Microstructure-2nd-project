import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
import ssl
import matplotlib.pyplot as plt
import seaborn as sns

# Set your OpenAI API key

openai.api_key = "Your API Key here"

# Load data
df = pd.read_csv("data/updated_data_2023.csv")

# Function to get text and file names by industry
def get_text_and_files_by_industry(input_industry):
    filtered_data = df[df['SIC'] == input_industry]
    if filtered_data.empty:
        raise ValueError("No files found for the given industry.")
    if len(filtered_data) < 10:
        raise ValueError("Less than 10 files found for the given industry.")
    return filtered_data['Text'].tolist()[:10], filtered_data['File'].tolist()[:10]

# Function to split text into chunks
def split_text_by_tokens(text, max_tokens=8192, model="text-embedding-ada-002"):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = tokenizer.encode(text)
    chunks = []
    
    # Split tokens into chunks
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)
        if chunk_text.strip():  # Ensure chunk is not empty
            chunks.append(chunk_text)
    
    return chunks

# Function to generate embeddings for a single chunk
def generate_embedding(text):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None  # Return None on failure

# Function to combine embeddings
def combine_embeddings(embeddings):
    valid_embeddings = [e for e in embeddings if e is not None]  # Ignore None values
    if not valid_embeddings:
        raise ValueError("No valid embeddings to combine.")
    return np.nanmean(valid_embeddings, axis=0).tolist()  # Use nanmean to handle NaN

# Process a single large text
def process_large_text(text):
    # Step 1: Split the text into chunks
    chunks = split_text_by_tokens(text, max_tokens=8191)  # Reserve 1 token for safety
    print(f"Split text into {len(chunks)} chunks.")

    # Step 2: Generate embeddings for each chunk
    chunk_embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Generating embedding for chunk {i+1}/{len(chunks)}...")
        chunk_embedding = generate_embedding(chunk)
        if chunk_embedding is not None:  # Only append valid embeddings
            chunk_embeddings.append(chunk_embedding)

    # Step 3: Combine chunk embeddings
    if not chunk_embeddings:
        raise ValueError("No valid embeddings were generated.")
    final_embedding = combine_embeddings(chunk_embeddings)
    return final_embedding

# Generate embeddings for multiple texts
def generate_embeddings_for_texts(text_list):
    embeddings = []
    for idx, text in enumerate(text_list):
        print(f"Processing text {idx + 1}/{len(text_list)}...")
        embedding = process_large_text(text)  # Process each large text
        embeddings.append(embedding)
    return embeddings

# Function to compute cosine similarity
def compute_cosine_similarity(embeddings):
    embedding_matrix = np.array(embeddings)
    similarity_matrix = cosine_similarity(embedding_matrix)
    return similarity_matrix

def plot_similarity_heatmap(similarity_matrix, labels):
    # Set figure dimensions and axes
    fig, ax = plt.subplots(figsize=(12, 12))  # Adjust size to fit labels and center
    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        cbar=True,
        ax=ax
    )
    
    # Set title and labels with padding
    ax.set_title("Cosine Similarity Heatmap", pad=20)
    ax.set_xlabel("Firms", labelpad=20)
    ax.set_ylabel("Firms", labelpad=20)
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    # Adjust layout
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.3)  # Add padding to center heatmap
    
    plt.show()

# Main execution
if __name__ == "__main__":
    try:
        # Get 10 texts and file names for a specific industry
        input_industry = "AGRICULTURE CHEMICALS [2870]" ## select the indusry you want to compare (> 10 companies)
        output_text_list, firm_labels = get_text_and_files_by_industry(input_industry)
        
        # Generate embeddings for all 10 texts
        print("Generating embeddings for all 10 texts...")
        embeddings = generate_embeddings_for_texts(output_text_list)

        # Calculate cosine similarity matrix
        print("Calculating cosine similarity matrix...")
        similarity_matrix = compute_cosine_similarity(embeddings)

        # Plot similarity heatmap
        plot_similarity_heatmap(similarity_matrix, firm_labels)
    
    except Exception as e:
        print(f"An error occurred: {e}")