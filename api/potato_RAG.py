import pinecone

pinecone.init(api_key="pcsk_2CDR3j_i4GsT4JQq2qq2pRsB9Pj5sJfFSBNtf8Fck6e6kRNegL6yJGHbV2ZvjahKXxoWs", environment="us-east-1")
index = pinecone.Index("potato-disease-index")

from openai import OpenAIEmbeddings

embed_model = OpenAIEmbeddings()

# Assume you have a list of documents
texts = ["Potato blight is caused by...", "Blackleg is a bacterial disease..."]

# Generate embeddings
vectors = embed_model.embed_documents(texts)

# Format and upsert to Pinecone
pinecone_vectors = [
    {"id": f"doc-{i}", "values": vec, "metadata": {"text": texts[i]}}
    for i, vec in enumerate(vectors)
]
index.upsert(pinecone_vectors)
