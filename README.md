AI PDF Chatbot

	A Streamlit-based AI assistant that lets you ask natural language questions from uploaded PDF documents. It uses Hugging Face Transformers for embeddings and Pinecone for semantic search.

Features

	1. Upload multiple PDF files.
	2. Extracts and chunks PDF text for processing.
	3. Embeds content using all-MiniLM-L6-v2 model.
	4. Stores embeddings in Pinecone vector DB.
	5. Accepts user queries via text.
	6. Generates answers using google/flan-t5-base.
	7. Shows the most relevant source chunks used for answering.

Tech Stack Used: 

	Frontend UI	- Streamlit
	PDF Text Extraction	- LangChain + PyPDFLoader
	Text Chunking	- LangChain's RecursiveCharacterTextSplitter
	Embedding Model	- sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
	Vector DB	- Pinecone
	LLM - google/flan-t5-base via HuggingFace Transformers
	Frameworks	- LangChain, Transformers, Streamlit

Project Structure

	├── app.py                 # Streamlit app
	├── utils.py               # Core logic for PDF processing, search, answer generation
	├── requirements.txt       # Python dependencies
	├── .env                   # API keys (excluded from git)
	├── sample            		 # Example PDF for testing
	├── README.md              # Project documentation

Setup Instructions

	1. Clone the repository
	2. Create virtual environment (optional but recommended)
	3. Install dependencies
	4. Create a .env file in the root directory
	5. Run the App Locally using 'streamlit run app.py'

Known Issues / Limitations

	1. Answers are generated even if document context is not confidently matched (no threshold filtering).
	2. Large PDFs might take time to process.
	3. Does not yet support voice input or page-level metadata.
	4. Whisper-based speech-to-text support was explored but removed due to reliability issues.

