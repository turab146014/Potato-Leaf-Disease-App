import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone

# Config
pinecone.init(api_key="pcsk_2CDR3j_i4GsT4JQq2qq2pRsB9Pj5sJfFSBNtf8Fck6e6kRNegL6yJGHbV2ZvjahKXxoWs", environment="us-east-1")
embedding_model = OpenAIEmbeddings(openai_api_key="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")

index_name = "potato-disease-index"

# Example data (You can also load from a text file or PDF)
texts = [
    "Early Blight is a common potato disease caused by Alternaria solani. Symptoms include dark spots on leaves with concentric rings...",
    "Late Blight is a severe disease caused by Phytophthora infestans. Symptoms include brown lesions and white mold under moist conditions...",
    "Healthy potatoes show no visible signs of infection. Leaves are green and free from lesions or spots..."
]

metadatas = [{"text": t} for t in texts]

# Create index if not exists
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)

# Embed and upsert
vectors = embedding_model.embed_documents(texts)
ids = [f"doc-{i}" for i in range(len(texts))]

to_upsert = list(zip(ids, vectors, metadatas))
index.upsert(vectors=to_upsert)
