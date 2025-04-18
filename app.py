# important libraries
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from flask import Flask, request
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

#created Flask application
app = Flask(__name__)
# Set the path to the database
db_path = "db/"
# Create the database directory if it doesn't exist
ollama = OllamaLLM(model="llama3.1")
# Create instance for the Ollama Embeddings
embeddings = OllamaEmbeddings(model="llama3.1")
#Create instance for the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)
# Create PromptTemplate for the raw prompt
raw_prompt = PromptTemplate.from_template(""" 
    <s>[INST] You are a technical assistant good at searching documents and answering questions. Hereby, the context of a Resume is provided answer the Question according to the Context and if you don't know the answer, say "I don't know".[/INST] </s>
    [INST] {input}
        Context: {context}
        Answer: 
    [/INST]""")                             

# Create an endpoint for asking any question
@app.route("/ai", methods=["POST"])
def aiPost():
    # Takes JSON as the request body
    json_content = request.json
    # Extract the query from the JSON content
    query = json_content.get("query")
    # Invoke Ollama Model from the query given
    response = ollama.invoke(query)
    # Get the response from the model
    response_answer = {"answer": response}
    # Return the response as JSON
    return response_answer

# Create an endpoint for asking questions about the PDF
@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    # Takes JSON as the request body
    json_content = request.json
    # Extract the query from the JSON content
    query = json_content.get("query")

    # Load the vector store from the database
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Get Chunks from the vector store with the similarity score threshold
    print("Creating retriever")
    retriever = vector_store.as_retriever(
        search_type = "similarity_score_threshold", 
        search_kwargs={
            "k": 20, 
            "score_threshold": 0.1,
        },
    )
    
    # Create a chain with prompt template and model
    document_chain = create_stuff_documents_chain(ollama, raw_prompt)
    # Feed the chunks to the model and prompt template
    chain = create_retrieval_chain(retriever, document_chain)
    # Invoke the model with the query
    result = chain.invoke({"input": query})
    print("result=", result)

    # Get relevant sources from the result
    sources = []
    # Loop through the context and get the page number and content
    for doc in result["context"]:
        sources.append({"Page Number":doc.metadata["page"], "Context":doc.page_content})    
    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

# Create an endpoint for uploading a PDF file
@app.route("/pdf", methods=["POST"])
def pdfPost():
    # Check if the request contains a file
    file = request.files['file']
    # Check if the file is empty
    file_name = file.filename
    # Save file in pdf/ directory
    save_file = "pdf/" + file_name
    file.save(save_file)
    print("File saved as: " + save_file)

    # Load the PDF file and split by pages
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print("Docs len=", len(docs))

    # Split the documents into chunks
    chunks = text_splitter.split_documents(docs)
    print("Chunks len=", len(chunks))

    # Create a vector store from the chunks and embeddings
    vectorestore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    vectorestore.persist()
    
    # Create a response by the server.
    response = {"status" : "success", "file_name": file_name, "message": "File saved successfully", "Doc Length": len(docs), "Chunks Length": len(chunks)}
    return response



def start_app():
    # Run the Flask app
    app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    start_app()