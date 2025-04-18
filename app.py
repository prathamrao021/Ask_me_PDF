from langchain_ollama import OllamaLLM, OllamaEmbeddings
from flask import Flask, request
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate


app = Flask(__name__)
db_path = "db/"
ollama = OllamaLLM(model="llama3.1")
embeddings = OllamaEmbeddings(model="llama3.1")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False)

raw_prompt = PromptTemplate.from_template(""" 
    <s>[INST] You are a technical assistant good at searching documents and answering questions. If you don't know the answer, say "I don't know".[/INST] </s>
    [INST] {input}
        Context: {context}
        Answer: 
    [/INST]""")                             

@app.route("/ai", methods=["POST"])
def aiPost():
    json_content = request.json
    query = json_content.get("query")
    
    response = ollama.invoke(query)
    response_answer = {"answer": response}
    return response_answer

@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    json_content = request.json
    query = json_content.get("query")

    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)

    print("Creating retriever")
    retriever = vector_store.as_retriever(
        search_type = "similarity_score_threshold", 
        search_kwargs={
            "k": 20, 
            "score_threshold": 0.1,
        },
    )
    
    document_chain = create_stuff_documents_chain(ollama, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})
    print("result=", result)

    sources = []
    for doc in result["context"]:
        sources.append({"Page Number":doc.metadata["page"], "Context":doc.page_content})    
    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer

@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files['file']
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print("File saved as: " + save_file)

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print("Docs len=", len(docs))

    chunks = text_splitter.split_documents(docs)
    print("Chunks len=", len(chunks))

    vectorestore = Chroma.from_documents(chunks, embeddings, persist_directory=db_path)
    vectorestore.persist()
    
    response = {"status" : "success", "file_name": file_name, "message": "File saved successfully", "Doc Length": len(docs), "Chunks Length": len(chunks)}
    return response



def start_app():
    app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    start_app()