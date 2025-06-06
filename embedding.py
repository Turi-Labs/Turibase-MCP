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
    Create or update embeddings for Markdown files incrementally.
    Only processes new or modified files since last run.

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

    # Load existing embeddings if they exist
    existing_data = {}
    file_timestamps = {}
    
    if os.path.exists(embeddings_path):
        logger.info(f"Loading existing embeddings from {embeddings_path}")
        try:
            with open(embeddings_path, 'rb') as f:
                existing_data = pickle.load(f)
            file_timestamps = existing_data.get('file_timestamps', {})
            logger.info(f"Loaded {len(existing_data.get('texts', []))} existing text chunks")
        except Exception as e:
            logger.warning(f"Could not load existing embeddings: {e}. Starting fresh.")
            existing_data = {}

    # Initialize data structures
    all_texts = existing_data.get('texts', [])
    all_metadatas = existing_data.get('metadatas', [])
    all_embeddings = existing_data.get('embeddings', np.array([])).tolist() if len(existing_data.get('embeddings', [])) > 0 else []

    supported_extensions = (".md",)

    # Ensure directory exists
    if not os.path.exists(directory):
        logger.error(f"Directory {directory} does not exist")
        raise ValueError(f"Directory {directory} does not exist")

    # Get current files and their modification times
    current_files = {}
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(supported_extensions):
                full_path = os.path.join(root, filename)
                rel_path = os.path.relpath(full_path, directory)
                current_files[rel_path] = os.path.getmtime(full_path)

    if not current_files:
        logger.warning(f"No markdown files found in directory {directory}")
        return len(all_texts) if all_texts else 0

    # Find files that need processing (new or modified)
    files_to_process = []
    for rel_path, mod_time in current_files.items():
        if rel_path not in file_timestamps or file_timestamps[rel_path] < mod_time:
            files_to_process.append(rel_path)

    # Remove embeddings for files that no longer exist
    files_to_remove = set(file_timestamps.keys()) - set(current_files.keys())
    if files_to_remove:
        logger.info(f"Removing embeddings for {len(files_to_remove)} deleted files")
        # Create new lists without the removed files
        new_texts = []
        new_metadatas = []
        new_embeddings = []
        
        for i, metadata in enumerate(all_metadatas):
            if metadata['source'] not in files_to_remove:
                new_texts.append(all_texts[i])
                new_metadatas.append(all_metadatas[i])
                new_embeddings.append(all_embeddings[i])
        
        all_texts = new_texts
        all_metadatas = new_metadatas
        all_embeddings = new_embeddings
        
        # Remove from timestamps
        for file_path in files_to_remove:
            del file_timestamps[file_path]

    # Remove embeddings for modified files (they'll be re-added)
    modified_files = [f for f in files_to_process if f in file_timestamps]
    if modified_files:
        logger.info(f"Updating embeddings for {len(modified_files)} modified files")
        # Remove old embeddings for modified files
        new_texts = []
        new_metadatas = []
        new_embeddings = []
        
        for i, metadata in enumerate(all_metadatas):
            if metadata['source'] not in modified_files:
                new_texts.append(all_texts[i])
                new_metadatas.append(all_metadatas[i])
                new_embeddings.append(all_embeddings[i])
        
        all_texts = new_texts
        all_metadatas = new_metadatas
        all_embeddings = new_embeddings

    if not files_to_process:
        logger.info("No new or modified files to process")
        return len(all_texts)

    logger.info(f"Processing {len(files_to_process)} new/modified files")

    # Process new/modified files
    for filename in tqdm(files_to_process, desc="Processing files"):
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

            # Update timestamp
            file_timestamps[filename] = current_files[filename]
            logger.info(f"Successfully processed {filename}")

        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            continue

    # Check if we have any valid texts
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
        "index": faiss.serialize_index(index),
        "file_timestamps": file_timestamps  # Store file modification times
    }

    # Ensure the directory for embeddings exists
    os.makedirs(os.path.dirname(embeddings_path) or '.', exist_ok=True)

    logger.info(f"Saving embeddings to {embeddings_path}...")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(data, f)

    logger.info("Embeddings saved successfully!")
    return len(all_texts)



# todo: Currently the code can create a new embedding file. but i want to add a function to update the existing embedding file.


create_embeddings_for_files("knowledgebase")