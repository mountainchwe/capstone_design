# function : to check records of students' face vectors in the chromaDB

import chromadb

client = chromadb.PersistentClient("./chroma_db")
collection = client.get_collection("face_vector") # face_vector = our table name

registered = collection.get(include=["metadatas","embeddings"]) # id(default) + metadata(name) + embeddings(facevectors)
print(registered)