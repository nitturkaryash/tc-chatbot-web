import os
from dotenv import load_dotenv
import chromadb
import google.generativeai as genai
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=gemini_api_key, transport="rest")

google_ef = GoogleGenerativeAiEmbeddingFunction(
    model_name="models/text-embedding-004",
    task_type="RETRIEVAL_DOCUMENT",
)

chroma_client = chromadb.PersistentClient(path="./chroma_persistent_storage")
collection_name = "document_qa_collection_gemini"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=google_ef
)

model = genai.GenerativeModel("models/gemini-1.5-flash")
response = model.generate_content("Hello world")

print(response.text)

