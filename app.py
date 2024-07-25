from flask import Flask, render_template, request, jsonify
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os
from threading import Thread

app = Flask(__name__)

# This is a simplified version, you'll need to adapt it to your specific setup
def initialize_bot():
    # Load and process the PDF
    file_path = "general.pdf"
    data_file = UnstructuredPDFLoader(file_path)
    docs = data_file.load()

    # Split documents and create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # Initialize embeddings
    HF_TOKEN = "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"
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )

    # Create vector store
    vectorstore = Chroma.from_documents(chunks, embeddings)
    vectorstore_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Create keyword retriever
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = 3

    # Create ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever],
        weights=[0.5, 0.5]
    )

    # Initialize LLM
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.3, "max_new_tokens": 1024},
        huggingfacehub_api_token=HF_TOKEN,
    )

    # Create prompt template
    template = """
    CONTEXT: {context} </s>

    QUERY: {query} </s>

    INSTRUCTIONS: - Use only the information provided in the CONTEXT section to answer the QUERY.
                  - Do not provide information or answers outside of the given CONTEXT.
                  - Provide only the answer to the query without additional information.
    ANSWER: The answer to the query is:
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    # Create the chain
    chain = (
        {"context": ensemble_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )

    return chain

# Initialize the bot
bot_chain = initialize_bot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query_bot():
    user_query = request.json['query']
    raw_response = bot_chain.invoke(user_query)
    answer = extract_answer(raw_response)
    return jsonify({'response': answer})

def extract_answer(response):
    parts = response.split('ANSWER:')
    if len(parts) > 1:
        return parts[1].strip()
    return response.strip()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)