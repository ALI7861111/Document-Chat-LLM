from flask import Flask, render_template, request, jsonify, session
import PyPDF2
from io import BytesIO

from langchain import text_splitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings , HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA , ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import os
# Load environment variables from .env file
load_dotenv()


memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
# Access the Hugging Face token from the environment
hugging_face_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
open_ai_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
app.secret_key = "M_x&n?FlE%qvV6?MEas2MQ#$"  # Change this to a secure random key
app.config['SECRET_KEY'] = "M_x&n?FlE%qvV6?MEas2MQ#$"



@app.route('/')
def index():
    return render_template('upload.html')

############### Chat Bot Functions #########################


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks , embedding_type):
    if embedding_type == 'openai':
        embeddings = OpenAIEmbeddings()
    elif embedding_type == 'huggingface':
        model_name = "hkunlp/instructor-large"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore, embedding_type):
    if embedding_type == 'openai':
        llm = ChatOpenAI()
    elif embedding_type == 'huggingface':
    #memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
        llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    conversation_chain = ConversationalRetrievalChain.from_llm(
                            llm,
                            retriever=vectorstore.as_retriever(),
                            memory=memory
                            )

    return conversation_chain



############ Flask App Functions #################################


@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist('pdf_files')
    pdf_text = ""

    for pdf_file in uploaded_files:
        if pdf_file and pdf_file.filename.endswith('.pdf'):
            print(f"Processing file: {pdf_file.filename}")
            pdf_content = pdf_file.read()  # Read the content of the file
            pdf_stream = BytesIO(pdf_content)  # Create a BytesIO stream from the content
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
        else:
            print(f"Skipped {pdf_file.filename} due to invalid format or no file uploaded.")

    # Store pdf_text in the session
    session['pdf_text'] = pdf_text

    print("PDF texts extracted successfully.")
    return render_template('upload.html')

@app.route('/ask')
def ask():
    embedding_choice = request.args.get('embedding', 'openai')
    question = request.args.get('question', '')
    pdf_text = session.get('pdf_text', '')# 
    # print(question)
    # print(pdf_text)
    chunks = get_text_chunks(pdf_text)
    vectorstore = get_vectorstore(chunks, embedding_type=embedding_choice)
    qa_chain = get_conversation_chain(vectorstore, embedding_type=embedding_choice)
    result = qa_chain({"question": question})
    answer = result["answer"]
    if 'chat_history' not in session:
        session['chat_history'] = []
    else:
        session['chat_history'].extend([(question, 'user'), (answer, 'bot')])
    return jsonify({'answer': answer})

@app.route('/fetch_webpage')
def fetch_webpage():
    url = request.args.get('url', '')
    if not url:
        return jsonify({'status': 'error', 'message': 'URL not provided'})

    try:
        # Fetch webpage content
        response = requests.get(url)
        response.raise_for_status()  # Check if request was successful
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text content
        webpage_text = ''.join(soup.stripped_strings)
        session['pdf_text'] = webpage_text
        print(session['pdf_text'])
        return jsonify({'status': 'success', 'message': 'Webpage content fetched successfully'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    answer = ""
    if request.method == 'POST':
        embedding_choice = request.args.get('embedding', 'openai')
        url = request.form.get('webpage_url', '')
        question = request.form.get('question', '')
        # Fetch and process webpage content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        webpage_text = ''.join(soup.stripped_strings)
        chunks = get_text_chunks(webpage_text)
        vectorstore = get_vectorstore(chunks, embedding_type=embedding_choice)
        qa_chain = get_conversation_chain(vectorstore, embedding_type=embedding_choice)
        result = qa_chain({"question": question})
        answer = result["answer"]

        if 'chat_history' not in session:
            session['chat_history'] = []
        else:
            session['chat_history'].extend([(question, 'user'), (answer, 'bot')])
    
        # ... your processing logic for the question...

    return render_template('upload.html', answer=answer)

@app.route('/erase_chat_history', methods=['GET', 'POST'])
def erase_chat_history():
    if request.method == 'POST':
        session['chat_history'] = []
    return render_template('upload.html', answer='Chat History cleared')


if __name__ == '__main__':
    app.run(debug=True)
