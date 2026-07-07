"""
Construction de la base vectorielle à partir du guide touristique PDF.
Utilise HuggingFace pour les embeddings (gratuit, local, pas de clé API).
Groq sera utilisé plus tard uniquement pour le LLM (dans graph.py).
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


def build_vectorstore(pdf_path: str = "../data/guide_tourisme_maroc.pdf") -> InMemoryVectorStore:
    """
    Charge le PDF, le découpe en chunks, calcule les embeddings (HuggingFace,
    pas OpenAI) et retourne un InMemoryVectorStore prêt à être interrogé.
    """
    print("Chargement du PDF...")
    loader = PyPDFLoader(pdf_path)
    data = loader.load()
    print(f"Loaded {len(data)} pages from the PDF document.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)
    print(f"Documents découpés en {len(all_splits)} chunks")

    # IMPORTANT : embeddings HuggingFace (local, gratuit) -> AUCUNE clé API requise
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = InMemoryVectorStore(embeddings)
    ids = vector_store.add_documents(documents=all_splits)
    print(f"Added {len(ids)} documents to the vector store.")

    return vector_store


if __name__ == "__main__":
    vs = build_vectorstore()
    results = vs.similarity_search(
        "Quels sont les boissons et pâtisseries incontournables au Maroc ?"
    )
    print(f"\nTrouvé {len(results)} document(s) similaire(s).")
    print(results[0].page_content[:300])