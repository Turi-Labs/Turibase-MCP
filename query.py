import os
import pickle
import numpy as np
import faiss
from openai import OpenAI
from dotenv import load_dotenv

def rag_base(question: str, embeddings_path: str) -> str:
    try:
        # Load environment variables
        load_dotenv()
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Load the saved embeddings
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)

        texts = data["texts"]
        metadatas = data["metadatas"]
        stored_embeddings = data["embeddings"]

        # Deserialize the FAISS index
        index = faiss.deserialize_index(data["index"])

        # Generate embedding for the question
        response = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        )
        query_embedding = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)

        # Search the index
        k = 5  # Number of results to retrieve
        distances, indices = index.search(query_embedding, k)

        # Prepare context from retrieved documents
        context = ""
        sources = []

        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                context += texts[idx] + "\n\n"
                sources.append(metadatas[idx])

        # Generate answer using OpenAI
        prompt = f"""
        Answer the following question based on the provided context:

        Context:
        {context}

        Question: {question}

        Answer:
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = completion.choices[0].message.content

        # Format the response
        response = f"Question: {question}\n\n"
        response += f"Answer: {answer}\n\n"
        response += "Sources:\n"
        for i, source in enumerate(sources, 1):
            response += f"{i}. {source.get('source', 'Unknown source')}\n"
        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
# print(rag_base("is there an language model which runs on diffusion?", "embeddings.pkl"))