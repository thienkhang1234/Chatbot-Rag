from langchain_community.vectorstores import Chroma

def create_vector_db(chunks, embedding_model, persist_directory="./db"):
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectordb.as_retriever(search_kwargs={"k": 3})
