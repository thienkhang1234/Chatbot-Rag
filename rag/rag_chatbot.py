from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def get_rag_chain(retriever):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
