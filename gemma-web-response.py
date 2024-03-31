from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

ollama = Ollama(model='gemma')

loader = WebBaseLoader('https://en.wikipedia.org/wiki/Lal_Bahadur_Shastri')
loader.requests_kwargs = {'verify':False}
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(all_splits, embeddings)

query = "How many countries were visisted by Lal Bahadur Shastri during his tenure of prime minister of India?"
docs = vectorstore.similarity_search(query) 
print(docs[0].page_content)
