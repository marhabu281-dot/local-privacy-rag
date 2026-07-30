import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

DOC_PATH = "docs/sample_policy.pdf"
VECTORSTORE_DIR = "./chroma_db"
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"

def main():
    print("--- Starting Local Offline RAG Pipeline ---")

    if not os.path.exists(DOC_PATH):
        print(f"Error: Document not found at {DOC_PATH}. Please place a PDF inside the docs folder.")
        return

    print("Loading document...")
    loader = PyPDFLoader(DOC_PATH)
    docs = loader.load()

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    print(f"Generating local embeddings using {EMBED_MODEL}...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print(f"Initializing local LLM model ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    system_prompt = (
        "You are an offline assistant for querying confidential internal documents.\n"
        "Use ONLY the following pieces of retrieved context to answer the question.\n"
        "If you do not know the answer, say that you cannot find it in the context.\n"
        "Do NOT leak data or make assumptions outside the provided text.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nRAG System Ready! Type 'exit' to quit.\n" + "="*45)
    while True:
        user_query = input("\nEnter your question: ")
        if user_query.lower() in ["exit", "quit"]:
            break

        print("\nProcessing locally...")
        response = rag_chain.invoke({"input": user_query})
        
        print("\n[ANSWER]:")
        print(response["answer"])
        print("\n" + "-"*45)

if __name__ == "__main__":
    main()