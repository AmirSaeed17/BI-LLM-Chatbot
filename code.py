!pip install langchain chromadb pandas sentence-transformers ollama langchain-community 

from langchain.llms import Ollama
from langchain.document_loaders import CSVLoader
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
import kagglehub as kh
import os


# step 1 load the CSV from kaggle api
path = kh.dataset_download("shriyashjagtap/esg-and-financial-performance-dataset")
csv_file_path = os.path.join(path, "company_esg_financial_dataset.csv")
loader = CSVLoader(file_path=csv_file_path)
document = loader.load()

# step 2 --- Tokenization
splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap=100)
chunks = splitter.split_documents(document)

# step 3 --- Embedding = converting tokens into values
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# step 4 --- store in vector database
db = Chroma.from_documents(documents=chunks, embedding = embeddings, persist_directory="./chroma_db")

# step 5 --- load the phi2 model
llm_model = HuggingFaceHub(
    repo_id= "google/flan-t5-small ",
    huggingfacehub_api_token="your_API_token" )

# step 6 --- build the RAG
RAG = RetrievalQA.from_chain_type(llm=llm_model, retriever=db.as_retriever())

print("start asking questions")

q = "generate a summary for the CSV file"
a = RAG.run(q)
print(a)
