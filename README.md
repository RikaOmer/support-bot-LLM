# Support Bot LLM

A RAG (Retrieval-Augmented Generation) chatbot for technical support, built with LangChain, Ollama, and ChromaDB.

## Features

- 🤖 **RAG-based Q&A**: Answers questions using your technical documentation
- 🔍 **Smart Document Retrieval**: Uses MMR (Maximum Marginal Relevance) for diverse, relevant results
- 🌐 **Multilingual Support**: Optimized for Hebrew with multilingual embeddings
- 📚 **Markdown Processing**: Converts HTML/Word docs to Markdown and intelligently chunks them
- 🎯 **Context-Aware**: Preserves document structure and headers for better context

## Architecture

1. **convert_html_to_md.py**: Converts HTML/Word documents to Markdown format
2. **build_db.py**: Builds a vector database from Markdown files using intelligent chunking
3. **rag_bot.py**: Interactive chatbot that retrieves and answers questions

## Setup

### Prerequisites

- Python 3.8+
- Ollama installed and running
- Required Python packages (see below)

### Installation

1. Install dependencies:
```bash
pip install langchain-chroma langchain-ollama langchain-core langchain-huggingface langchain-text-splitters markdownify
```

2. Install and run Ollama models:
```bash
# Install a lightweight LLM model
ollama pull llama3.1
# Or use: mistral:7b, phi3:mini, llama3.2:3b
```

3. Prepare your data:
   - Place HTML/Word documents in the `wordFilter/` directory
   - Run `convert_html_to_md.py` to convert them to Markdown
   - Markdown files will be saved in `markdown/` directory

4. Build the vector database:
```bash
python build_db.py
```

5. Run the chatbot:
```bash
python rag_bot.py
```

## Configuration

### Models

Edit the model settings in `rag_bot.py` and `build_db.py`:

**LLM Models (Ollama):**
- `llama3.1` - Excellent quality (default)
- `mistral:7b` - Great quality, very efficient
- `phi3:mini` - Very lightweight (~3.8B)
- `llama3.2:3b` - Ultra-lightweight

**Embedding Models:**
- `intfloat/multilingual-e5-small` - Lightweight (~130MB), great for Hebrew (default)
- `paraphrase-multilingual-mpnet-base-v2` - Better quality (~420MB)

### Paths

Update paths in `build_db.py` and `convert_html_to_md.py` to match your directory structure.

## Usage

1. Start the bot: `python rag_bot.py`
2. Ask questions in Hebrew or English
3. The bot will search the database and provide answers based on your documentation
4. Type `exit` or `יציאה` to quit

## How It Works

1. **Document Processing**: HTML/Word docs are converted to Markdown
2. **Smart Chunking**: Documents are split by headers first, then by size (600 chars with 150 overlap)
3. **Vector Embedding**: Chunks are embedded using multilingual models
4. **Retrieval**: MMR search finds diverse, relevant document chunks
5. **Generation**: LLM generates answers based on retrieved context

## Troubleshooting

- **Model not found**: Make sure Ollama is running and the model is installed (`ollama list`)
- **Database not found**: Run `build_db.py` first
- **Wrong answers**: Rebuild the database after changing embedding models
- **Poor retrieval**: Adjust `k` and `fetch_k` parameters in `rag_bot.py`

## License

MIT
