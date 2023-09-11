from PyPDF2 import PdfReader 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS 
#from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader , CSVLoader 
from langchain.llms import OpenAI
from dotenv import load_dotenv
import openai
import os

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']



#For loading csv files 
#csv_loader = CSVLoader(file_path="./docs.csv")


def pdf_to_faiss(file_path):
    # Load PDF File from the file path
    pdf_reader = PdfReader(file_path)


    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def text_to_faiss(file_path):
    #Load Text file
    text_reader = TextLoader(file_path)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(text_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    # Splitting up the text into smaller chunks for indexing
    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200, #striding over the text
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch


def csv_to_faiss(file_path):
    #Load Text file
    csv_loader = CSVLoader(file_path,csv_args={"delimiter": ","})

    # read data from the file and put them into a variable called raw_text
    # raw_text = ''
    # for i, page in enumerate(csv_reader.pages):
    #     text = page.extract_text()
    #     if text:
    #         raw_text += text

    # # Splitting up the text into smaller chunks for indexing
    # csv_splitter = CharacterTextSplitter(        
    #     separator = "\n",
    #     chunk_size = 100,
    #     chunk_overlap  = 20, #striding over the text
    #     length_function = len,
    # )
    # texts = csv_splitter.split_text(raw_text)

    data = csv_loader.load()

    # Download embeddings from OpenAI
    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_documents(data, embeddings)
    return docsearch

def main():
    file_path = "FK_ads.csv"
    doc_faiss = csv_to_faiss(file_path)
    doc_faiss.save_local("fk_1000")



if __name__ == "__main__":
    main()
