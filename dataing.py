import os
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS index
embedding_size = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(embedding_size)

# Store document metadata
documents = []

def load_documents(path):
    texts = []
    filenames = []

    if os.path.isdir(path):  # If it's a folder, load all .txt files
        for file in os.listdir(path):
            if file.endswith(".txt"):
                file_path = os.path.join(path, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    texts.append(text)
                    filenames.append(file)
    elif os.path.isfile(path) and path.endswith(".txt"):  # If it's a single .txt file
        with open(path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            filenames.append(os.path.basename(path))
    else:
        raise ValueError("❌ Invalid path! Provide a folder or a .txt file.")

    return texts, filenames

# Path to your documents (change this to a folder or file path)
document_path = "/mnt/igi_mark_prisbot/prismatic_chatbot_script.txt"  # Or "./pris.txt" if it's a single file

# Load and process documents
docs, filenames = load_documents(document_path)
doc_embeddings = model.encode(docs, convert_to_numpy=True)  # Generate embeddings

# Convert embeddings to NumPy array and add to FAISS
index.add(doc_embeddings)

# Save FAISS index & metadata
faiss.write_index(index, "faiss_index.bin")
with open("documents.pkl", "wb") as f:
    pickle.dump({"texts": docs, "filenames": filenames}, f)

print(f"✅ Indexed {len(docs)} document(s) in FAISS!")
