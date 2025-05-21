import os
import pickle
from tqdm import tqdm
import faiss
import numpy as np
from openai import OpenAI  # Updated import
import logging
from dotenv import load_dotenv

def create_embeddings_for_files(directory, embeddings_path="embeddings.pkl"):
    """
    Create embeddings for Markdown files and store them locally without LangChain.

    :param directory: Path to the directory containing files
    :param embeddings_path: Path to store the embeddings
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Initialize the OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # For chunking text
    def split_text(text, chunk_size=1000, chunk_overlap=200):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text) and end - start == chunk_size:
                # Find the last space within the chunk to avoid cutting words
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            chunks.append(text[start:end])
            start = end - chunk_overlap if end - start > chunk_overlap else end
        return chunks

    all_texts = []
    all_metadatas = []
    all_embeddings = []

    # Changed to focus on Markdown files
    supported_extensions = (".md",)

    # Ensure directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        raise ValueError(f"Directory {directory} does not exist")

    # Get list of files
    files = [f for f in os.listdir(directory) if f.lower().endswith(supported_extensions)]

    if not files:
        logger.warning(f"No markdown files found in directory {directory}")
        return None

    logger.info(f"Found {len(files)} markdown files to process")

    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(directory, filename)
        try:
            # Read markdown file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():  # Skip empty documents
                logger.warning(f"Skipping empty file: {filename}")
                continue

            chunks = split_text(text)
            if chunks:  # Only add if we have valid chunks
                logger.debug(f"Processing {len(chunks)} chunks from {filename}")
                for chunk in chunks:
                    # Get embedding from OpenAI using the new API format
                    response = client.embeddings.create(
                        input=chunk,
                        model="text-embedding-ada-002"
                    )
                    embedding = response.data[0].embedding

                    metadata = {"source": filename}
                    all_metadatas.append(metadata)
                    all_texts.append(chunk)
                    all_embeddings.append(embedding)

            logger.info(f"Successfully processed {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue

    # Check if we have any valid texts before creating the index
    if not all_texts:
        logger.error("No valid text chunks were extracted from the documents")
        raise ValueError("No valid text chunks were extracted from the documents")

    logger.info(f"Creating FAISS index for {len(all_texts)} text chunks...")

    # Convert embeddings to numpy array
    embeddings_array = np.array(all_embeddings).astype('float32')

    # Create FAISS index
    dimension = len(all_embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    # Create a dictionary to store everything
    data = {
        "texts": all_texts,
        "metadatas": all_metadatas,
        "embeddings": embeddings_array,
        "index": faiss.serialize_index(index)
    }

    # Ensure the directory for embeddings exists
    os.makedirs(os.path.dirname(embeddings_path) or '.', exist_ok=True)

    logger.info(f"Saving embeddings to {embeddings_path}...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)

    logger.info("Embeddings saved successfully!")
    return len(all_texts)

# create_embeddings_for_files("knowledgebase/2025-05-20")