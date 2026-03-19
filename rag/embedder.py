from langchain_openai import OpenAIEmbeddings # Or HuggingFaceEmbeddings

def get_embedding_model():
    return OpenAIEmbeddings(model="text-embedding-3-small")
