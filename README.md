# Ask_me_PDF
Ask_me_PDF is a Flask-based application integrated with the LangChain framework to provide AI-powered document processing and question-answering capabilities. The application allows users to upload PDF documents, process them into manageable chunks, and store them in a vector database for efficient retrieval. Users can then query the documents, and the system leverages advanced language models to provide context-aware answers.

## Features
- **PDF Upload and Processing**: Upload PDF files, split them into smaller chunks, and store them in a vector database.
- **AI-Powered Question Answering**: Ask questions about the uploaded documents, and get accurate, context-aware answers.
- **Vector Database Integration**: Uses Chroma as a vector store to enable efficient similarity-based retrieval of document chunks.
- **Embeddings with Ollama**: Generates semantic embeddings using the `llama3.1` model for accurate document representation.
- **Customizable Prompt Templates**: Utilizes LangChain's prompt templates to guide the AI model's responses.

## Endpoints
1. **`/pdf`**: Upload a PDF file, process it into chunks, and store it in the vector database.
2. **`/ask_pdf`**: Query the uploaded PDF documents and retrieve context-aware answers.
3. **`/ai`**: General-purpose endpoint for interacting with the AI model.

## To Run This on Your Machine:
1. **Install Dependencies**:
   - Ensure you have Python installed on your system.
   - Create a virutal enviornment
   ```bash
   python3 -m venv .venv
   ```
   - Activate the virtual environment:
     - On Windows:
       ```bash
       .venv\Scripts\activate
       ```
     - On macOS/Linux:
       ```bash
       source .venv/bin/activate
       ```
   - Install the required Python libraries by running:
     ```bash
     pip install -r requirements.txt
     ```

2. **Download the Llama3.1 Model**:
   - Visit the [Ollama website](https://ollama.com) and download the `llama3.1` model to your local computer.

3. **Set Up the Application**:
   - Clone or download this repository to your local machine.
   - Ensure the `db/` directory exists for storing the vector database.
   - Ensure the `pdf/` directory exists for saving PDFs.

4. **Run the Flask Application**:
   - Start the Flask server by running:
     ```bash
     python app.py
     ```
   - The application will be available at `http://localhost:8080`.

5. **Use the Endpoints**:
   - Use tools like Postman or `curl` to interact with the endpoints:
     - `/pdf`: Upload a PDF file.
        - Example for `curl`:
            ```bash
            curl -X POST http://localhost:8080/pdf
            -H "Content-Type: multipart/form-data"
            -F "file=<pdf_path>"
            ```
     - `/ask_pdf`: Query the uploaded PDF.
        - Example for `curl`:
            ```bash
            curl -X POST http://localhost:8080/ask_pdf \
            -H "Content-Type: application/json" \
            -d '{"query": "<question_related_to_pdf_to_be_asked>"}'
            ```
     - `/ai`: General-purpose AI interaction.
        - Example for `curl`:
            ```bash
            curl -X POST http://localhost:8080/ai \
            -H "Content-Type: application/json" \
            -d '{"query": "<question>"}'
            ```

## Technologies Used
- **Flask**: Backend framework for building RESTful APIs.
- **LangChain**: Framework for building applications powered by language models.
- **Chroma**: Vector database for storing and retrieving document embeddings.
- **Ollama LLM**: Language model for generating embeddings and answering queries.
- **PDFPlumber**: Library for extracting text from PDF documents.

## How It Works
1. **Upload**: Users upload a PDF document via the `/pdf` endpoint.
2. **Processing**: The document is split into smaller chunks using a text splitter, and embeddings are generated for each chunk.
3. **Storage**: The chunks and their embeddings are stored in a Chroma vector database.
4. **Querying**: Users can query the document via the `/ask_pdf` endpoint, and the system retrieves relevant chunks to provide an answer.
