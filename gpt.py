from langchain.embeddings.openai import OpenAIEmbeddings
from  langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os



os.environ["OPENAI_API_KEY"] = "Your Key"
loader = DirectoryLoader('Store',glob='**/*.txt')
docs = loader.load()

chart_text_splitter= CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

doc_texts=chart_text_splitter.split_documents(docs)
print(doc_texts)


opernAI_embeddings= OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

vstore=Chroma.from_documents(doc_texts,opernAI_embeddings)
model= VectorDBQA.from_chain_type(llm=OpenAI(), chain_type='stuff',vectorstore=vstore)


questions="what is machine learning"
model.run(questions)