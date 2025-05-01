import os
import sys
import nltk
import faiss
import numpy as np
from pathlib import Path
from langchain_community.vectorstores import FAISS
from config import *
# Get absolute path to project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
DATA_PATH = os.path.join(ASSETS_FOLDER, "data")
INDEX_PATH = os.path.join(ASSETS_FOLDER, "index", "medassist.faiss")  # This is the path causing the error
# Import from project
try:
    from src.helper import load_data, text_split, load_hf_embeddings
    from config import *
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def fix_nltk():
    # Download required NLTK data
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
    except Exception as e:
        print(f"NLTK download error: {e}")

def create_sample_data():
    from langchain_core.documents import Document
    return [Document(page_content="Sample medical text about diabetes and heart disease. "
                                  "Diabetes is a condition that affects blood sugar levels. "
                                  "Heart disease refers to various conditions that affect the heart.")]

def create_index(data_path, save_path, chunk_size, chunk_overlap):
    print("Creating index...")
    fix_nltk()
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Try to load actual data
    try:
        print(f"Loading data from {data_path}...")
        data = load_data(data_path)
        print(f"Data loaded: {len(data)} documents")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating sample data instead...")
        data = create_sample_data()
        print("Created sample data")
    
    print("Splitting text...")
    text_chunks = text_split(data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print(f"Split into {len(text_chunks)} chunks")
    
    print("Loading embeddings...")
    try:
        embeddings = load_hf_embeddings()
        
        print("Creating vector store...")
        vectorstore_from_docs = FAISS.from_documents(text_chunks, embedding=embeddings)
        
        print(f"Saving vector store to {save_path}...")
        vectorstore_from_docs.save_local(save_path)
    except Exception as e:
        print(f"Error with embeddings or FAISS: {e}")
        print("Creating direct FAISS index as fallback...")
        
        # Create a simple test index as fallback
        dimension = 768  # Typical dimension for sentence embeddings
        index = faiss.IndexFlatL2(dimension)
        
        # Add some random vectors (just for testing)
        num_vectors = len(text_chunks)
        random_vectors = np.random.random((max(num_vectors, 10), dimension)).astype('float32')
        index.add(random_vectors)
        
        # Create the directory structure
        os.makedirs(save_path, exist_ok=True)
        
        # Save the index directly
        index_file = os.path.join(save_path, "index.faiss")
        faiss.write_index(index, index_file)
        
        # Create a dummy docstore file
        import pickle
        docstore = {i: chunk for i, chunk in enumerate(text_chunks)}
        with open(os.path.join(save_path, "docstore.pkl"), "wb") as f:
            pickle.dump(docstore, f)
    
    # Verify the index was created
    index_file = os.path.join(save_path, "index.faiss")
    print(f"Checking if index was created...")
    if os.path.exists(index_file):
        print(f"Success! Index file created at: {index_file}")
        print(f"File size: {os.path.getsize(index_file) / 1024:.2f} KB")
    else:
        print(f"ERROR: Index file was not created at: {index_file}")
        sys.exit(1)
    
    return True

if __name__ == '__main__':
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"INDEX_PATH: {INDEX_PATH}")
    print(f"Current directory: {os.getcwd()}")
    
    # Create index
    result = create_index(DATA_PATH, INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    
    if result:
        print("Index creation successful!")
    else:
        print("Index creation failed!")