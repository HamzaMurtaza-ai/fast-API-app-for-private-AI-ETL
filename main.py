from fastapi import FastAPI
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io, os, tempfile

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

app = FastAPI(title="GDrive → LangChain → Ollama → Qdrant Pipeline")

# ---- CONFIG ----
SERVICE_ACCOUNT_FILE = "gdrivedata-473806-45522984c22d.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
OLLAMA_MODEL = "bge-m3:latest"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "locl_drive_embeddings"

# ---- GDRIVE AUTH ----
def authenticate_drive():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES
    )
    service = build("drive", "v3", credentials=creds)
    return service

# ---- FILE DOWNLOAD ----
def download_file_from_drive(file_id: str, file_name: str) -> str:
    service = authenticate_drive()
    request = service.files().get_media(fileId=file_id)
    file_path = os.path.join(tempfile.gettempdir(), file_name)
    with io.FileIO(file_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
    print(f"✅ File downloaded: {file_path}")
    return file_path

# ---- FILE LOADING ----
def load_document(file_path: str):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8", autodetect_encoding=True)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents from {file_path}")
    return docs

# ---- CHUNKING ----
def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size= 4000, chunk_overlap= 200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")
    return chunks

# ---- EMBEDDINGS + QDRANT STORE ----
def embed_and_store(chunks):
    embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url="http://localhost:11434")
    qdrant = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
    )
    print(f"✅ Stored {len(chunks)} embeddings into Qdrant collection '{COLLECTION_NAME}'")
    return True

# ---- FASTAPI ENDPOINT ----
@app.api_route("/process_drive_file/{file_id}", methods=["GET", "POST"])
async def process_drive_file(file_id: str):
    try:
        # Get file metadata
        service = authenticate_drive()
        metadata = service.files().get(fileId=file_id, fields="name, mimeType").execute()
        file_name = metadata["name"]

        print(f"\n🚀 Processing File: {file_name} ({metadata['mimeType']})")

        # Download file
        file_path = download_file_from_drive(file_id, file_name)

        # Extract content
        docs = load_document(file_path)

        # Chunking
        chunks = chunk_documents(docs)

        # Create embeddings and store in Qdrant
        embed_and_store(chunks)

        return {"status": "success", "file": file_name, "chunks": len(chunks)}

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}
