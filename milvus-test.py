from pymilvus import MilvusClient
from pymilvus import model
import numpy as np

client = MilvusClient("./milvus_demo.db")
if client.has_collection(collection_name="demo_collection"):
    client.drop_collection(collection_name="demo_collection")
client.create_collection(
    collection_name="demo_collection",
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)


docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

# This will download a small embedding model "paraphrase-albert-small-v2" (~50MB).
embedding_fn = model.DefaultEmbeddingFunction()

vectors = embedding_fn.encode_documents(docs)

data = [{"id": 1, "vector":vectors[0], "text": docs[i], "subject":"history"} for i in range(len(docs))]
# print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="demo_collection", data=data)

# print(res)
query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])

res = client.search(
    collection_name="demo_collection",
    data=query_vectors,
    limit = 2,
    output_fields=["text", "subject"]
)

print(res)
