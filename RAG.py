from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain_community.llms import Ollama
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


# load the document and split it into chunks
textloader = TextLoader("./documents/llama2paper.md")
documents = textloader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function)

retriever = db.as_retriever(search_kwargs={"k": 1})

llm = Ollama(model="AIresearcher:latest")

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)

res = rag_chain.invoke({"input": "What is the Table of Content for the paper 'Llama 2: Open Foundation and Fine-Tuned Chat Models'?"})
answer = res['answer']

res2 = rag_chain.invoke({"input": "How has Llama 2 improved model convergence speed during training?"})
answer2 = res['answer']


print(answer)
print(answer2)