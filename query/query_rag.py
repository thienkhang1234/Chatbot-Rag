from rag.embedder import get_embedding_model
from rag.retriever import Chroma
from rag.rag_chatbot import get_rag_chain

def ask_bot(question):
    # Load the existing DB
    embed_model = get_embedding_model()
    vector_db = Chroma(persist_directory="./db", embedding_function=embed_model)
    retriever = vector_db.as_retriever()

    # Get the RAG answer
    rag_chain = get_rag_chain(retriever)
    response = rag_chain.invoke(question)
    return response["result"]

if __name__ == "__main__":
    query = input("Ask a question about your PDF: ")
    print(f"Bot: {ask_bot(query)}")
