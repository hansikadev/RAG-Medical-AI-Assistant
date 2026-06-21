import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import traceback

load_dotenv()

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

PINECONE_ENV="us-east-1" 
PINECONE_INDEX_NAME="medicalindex"

os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY 

UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR,exist_ok=True)


# initialize pinecone instance
pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws",region=PINECONE_ENV)
existing_indexes = pc.list_indexes().names()


if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)


index=pc.Index(PINECONE_INDEX_NAME)

# load,split,embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_paths = [] #SAVE UPLOADED FILES

    #1. save uploaded files
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    for file_path in file_paths:
        #2. load pdf files
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        #3. split into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        #4. embed chunks
        texts = [chunk.page_content for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata = chunk.metadata.copy()
            metadata["text"] = chunk.page_content
            metadatas.append(metadata)
        ids = [f"{Path(file_path).stem}-{i}" for i in range(len(chunks))]


        print(f"🔍 Embedding {len(texts)} chunks...")
        embeddings = embed_model.embed_documents(texts) #this is the step where text is converted into vector embeddings

        #5. upsert to pinecone
        print("📤 Uploading to Pinecone...")
        with tqdm(total=len(embeddings), desc="Upserting to Pinecone") as progress:
            index.upsert(vectors=zip(ids, embeddings, metadatas))
            progress.update(len(embeddings))

        print(f"✅ Upload complete for {file_path}")

print("PINECONE_API_KEY =", bool(PINECONE_API_KEY))
print("GOOGLE_API_KEY =", bool(GOOGLE_API_KEY))

print("Starting vector store initialization...")

try:
    print("PINECONE_API_KEY exists:", bool(PINECONE_API_KEY))
    print("GOOGLE_API_KEY exists:", bool(GOOGLE_API_KEY))

    pc = Pinecone(api_key=PINECONE_API_KEY)

    print("Connected to Pinecone")

    spec = ServerlessSpec(cloud="aws", region=PINECONE_ENV)

    existing_indexes = pc.list_indexes()

    print("Indexes:", existing_indexes)

except Exception as e:
    print("ERROR DURING STARTUP:")
    print(e)
    traceback.print_exc()
    raise