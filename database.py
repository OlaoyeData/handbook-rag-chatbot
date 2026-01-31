from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

from dotenv import load_dotenv
load_dotenv()

#Configuration
DATA_PATH = R"data"
CHROMA_PATH = r"chroma_db"

#initiate the embeddings model
embeddings_model =  GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

#initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

#loading the PDF document
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

#splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

#Creating the chunks
chunks = text_splitter.split_documents(raw_documents)

#Creating unique ID's
uuids = [str(uuid4()) for _ in range(len(chunks))]

#adding chunks to the vector store
vector_store.add_documents(documents=chunks, ids=uuids)