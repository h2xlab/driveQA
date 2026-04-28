from langchain_community.embeddings import HuggingFaceEmbeddings

model_name="sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

def get_embedding(input: str):
    return embeddings.embed_query(input)