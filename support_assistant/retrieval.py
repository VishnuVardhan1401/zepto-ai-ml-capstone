from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"


# =========================================================
# EMBEDDING MODEL
# =========================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# CHROMADB
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="zepto_policy",
    metadata={
        "hnsw:space": "cosine"
    }
)


# =========================================================
# TASK 1 - BUILD INDEX
# =========================================================

def build_index():

    files = sorted(
        DOCS_DIR.glob("doc_*.txt")
    )

    if len(files) != 8:
        raise RuntimeError(
            f"Expected 8 documents, found {len(files)}"
        )

    documents = []
    ids = []
    metadatas = []

    for file in files:

        text = file.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            raise RuntimeError(
                f"Empty document: {file.name}"
            )

        document_id = file.stem

        # One document = one chunk
        chunk_id = (
            f"{document_id}_chunk_01"
        )

        documents.append(text)

        ids.append(chunk_id)

        metadatas.append(
            {
                "document_id": document_id,
                "filename": file.name
            }
        )

    # Create embeddings
    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    )

    # Store embeddings in ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )

    print("\n==============================================")
    print("          MODULE 3 - TASK 1 COMPLETE")
    print("==============================================")

    print(
        f"Documents loaded      : {len(documents)}"
    )

    print(
        f"Chunks created        : {len(ids)}"
    )

    print(
        f"Embeddings created    : {len(embeddings)}"
    )

    print(
        f"ChromaDB collection   : {collection.name}"
    )

    print(
        f"ChromaDB stored items : {collection.count()}"
    )

    print("\nStored chunk IDs:")

    for chunk_id in ids:
        print(
            f" - {chunk_id}"
        )

    print("\n==============================================")
    print("Task 1 finished successfully.")
    print("==============================================")


# =========================================================
# RETRIEVAL - USED BY TASK 3
# =========================================================

def retrieve(
    query: str,
    k: int = 3
):

    # If the collection is empty, build it.
    if collection.count() != 8:
        build_index()

    # Embed the query
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    # Top-k cosine similarity retrieval
    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for i in range(
        len(results["ids"][0])
    ):

        retrieved.append(
            {
                "id": results["ids"][0][i],

                "document_id":
                    results["metadatas"][0][i][
                        "document_id"
                    ],

                "document":
                    results["documents"][0][i],

                "distance":
                    float(
                        results["distances"][0][i]
                    )
            }
        )

    return retrieved


# =========================================================
# TASK 1 RUNS ONLY WHEN THIS FILE IS DIRECTLY EXECUTED
# =========================================================

if __name__ == "__main__":
    build_index()