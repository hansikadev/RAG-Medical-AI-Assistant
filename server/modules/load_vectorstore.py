import os
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Environment Variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_ENV = "us-east-1"
PINECONE_INDEX_NAME = "medicalindex"

if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Upload directory
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_pinecone_index():
    """
    Connect to Pinecone only when needed.
    This prevents Render startup issues.
    """

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is missing")

    print("Connecting to Pinecone...")

    pc = Pinecone(api_key=PINECONE_API_KEY)

    spec = ServerlessSpec(
        cloud="aws",
        region=PINECONE_ENV
    )

    existing_indexes = pc.list_indexes().names()

    print("Existing indexes:", existing_indexes)

    if PINECONE_INDEX_NAME not in existing_indexes:
        print("Creating Pinecone index...")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=spec
        )

        while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
            time.sleep(1)

    print("Connected to Pinecone index")

    return pc.Index(PINECONE_INDEX_NAME)


def load_vectorstore(uploaded_files):
    """
    Load PDFs, split into chunks,
    generate embeddings, and upload to Pinecone.
    """

    # Initialize only when upload endpoint is called
    index = get_pinecone_index()

    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    file_paths = []

    # Save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename

        with open(save_path, "wb") as f:
            f.write(file.file.read())

        file_paths.append(str(save_path))

    # Process each PDF
    for file_path in file_paths:

        print(f"Processing {file_path}")

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        texts = [chunk.page_content for chunk in chunks]

        metadatas = []

        for chunk in chunks:
            metadata = chunk.metadata.copy()
            metadata["text"] = chunk.page_content
            metadatas.append(metadata)

        ids = [
            f"{Path(file_path).stem}-{i}"
            for i in range(len(chunks))
        ]

        print(f"Embedding {len(texts)} chunks...")

        embeddings = embed_model.embed_documents(texts)

        print("Uploading embeddings to Pinecone...")

        vectors = list(zip(ids, embeddings, metadatas))

        with tqdm(
            total=len(vectors),
            desc="Upserting to Pinecone"
        ) as progress:

            index.upsert(vectors=vectors)
            progress.update(len(vectors))

        print(f"Upload complete for {file_path}")