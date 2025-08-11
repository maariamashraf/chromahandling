from bs4 import BeautifulSoup
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import os

# === SETTINGS ===
STORAGE_PATH = r"C:\Users\maria\Downloads\VectorSearch-main\VectorSearch-main\storage"
COLLECTION_NAME = "docs"
HTML_FOLDER = r"C:\Users\maria\Downloads\VectorSearch-main\VectorSearch-main\output\documents -  html"  # folder path

# 1. Create persistent ChromaDB client
client = PersistentClient(path=STORAGE_PATH)

# 2. Create/get collection
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(COLLECTION_NAME)

# 3. Load embedding model (load once to avoid repeated loading)
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Iterate through all HTML files in folder
for filename in os.listdir(HTML_FOLDER):
    if not filename.lower().endswith(".html"):
        continue  # skip non-HTML files
    
    file_path = os.path.join(HTML_FOLDER, filename)

    # Read HTML content
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    # Check if document already stored
    existing_docs = collection.get(where={"source": file_path})
    if existing_docs["ids"]:
        print(f"Document '{filename}' already stored. Skipping...")
        continue

    # Embed the text
    embedding = model.encode([text])

    # Store in ChromaDB
    doc_id = os.path.splitext(filename)[0]
    collection.add(
        ids=[doc_id],
        embeddings=embedding.tolist(),
        documents=[text],
        metadatas=[{"source": file_path}]
    )
    print(f"Document '{filename}' added successfully!")

# 5. Retrieve with embeddings and preview vector
results = collection.get(include=["embeddings", "documents", "metadatas"])
print("\n=== Current documents in DB ===")
for i, doc_id in enumerate(results["ids"]):
    vec = results["embeddings"][i]
    print(f"\nID: {doc_id}")
    print(f"Source: {results['metadatas'][i]['source']}")
    print(f"Document length: {len(results['documents'][i])} characters")
    print(f"Vector length: {len(vec)}")
    print(f"First 10 vector values: {vec[:10]}")
