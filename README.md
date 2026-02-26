# Web3 Semantic Search

The **Web3 Semantic Search Project** is a decentralized content retrieval system in which users upload content - such as images and blogs - that can later be retrieved by other users using natural-language search (via queries or prompts). It was developed on **Camp Network** - a Layer-1 blockchain built to modernize intellectual property (IP) infrastructure and power the next generation of AI Agents on verifiable IP.

This system utilizes a **Retrieval-Augmented Generation** (RAG) architecture to handle the retrieval process, employing several foundational models, such as **LLMs** and **VLMs**, in its integration. I enabled the system to be integrated into a web-based application as API endpoints using **Python FASTAPI**. 

This project stands to promote a world of **decentralization** and **ownership of intellectual property (IP)**, allowing content (blogs, images, nft pfp) to be owned and controlled by its rightful owners, while also implementing an incentive-based model to reward creators.

---

## Tech Stack
<div align="left">

| Layer | Technology |
|-------|------------|
| **API** | [FastAPI](https://fastapi.tiangolo.com/) with [Uvicorn](https://www.uvicorn.org/) |
| **Vector database** | [ChromaDB](https://www.trychroma.com/) (persistent, local, cloud) |
| **AI: LLM & VLM** | [Groq](https://groq.com/) - Llama 3.1 8B (text), Llama 4 Scout 17B (vision) |
| **Image handling** | [Pillow](https://pillow.readthedocs.io/) (PIL) |
| **Config** | [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variables |

</div>

---

## Prerequisites

- **Python** 3.10 or higher
- **Groq API key**: [Create one](https://console.groq.com/) to use the LLM and vision endpoints.
- **Chroma API key** (optional): for [Chroma Cloud](https://www.trychroma.com/) storage instead of local persistence.

---

## To Setup or replicate project: 

1. **Clone the repository**
   ```bash
   git clone https://github.com/TolanSilas/web3-semantic-search.git

   cd web3-semantic-search
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment variables**  
   Create a `.env` file in the project root:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   CHROMA_DIR=./chroma_data
   ```

   - `GROQ_API_KEY` -  **Required.** Used for image description, text description, tag extraction, and query expansion.
   - `CHROMA_DIR` - Optional. Directory for ChromaDB persistence (default: `./chroma_data`). You can add your `CHROMA_API_KEY` to the `.env` file to utilize Chroma Cloud for storage.

5. **Run the API**
   ```bash
   python -m app.main
   ```
   Or with Uvicorn directly:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

   The API will be available at `http://127.0.0.1:8000`. Interactive docs: `http://127.0.0.1:8000/docs`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/analyze` | Analyze image or text before minting; returns enhanced description and tags |
| `POST` | `/index` | Store metadata in ChromaDB after minting (by `token_id`) |
| `POST` | `/search` | Natural-language search; returns matching token IDs and metadata |

---


## Contributing

Contributions are welcome. To contribute:

1. Fork the repository.
2. Create a branch for your change (`git checkout -b feature/your-feature` or `fix/your-fix`).
3. Commit your changes with clear messages.
4. Push to your fork and open a Pull Request against the default branch.

Please ensure your code and commits align with the project’s style and purpose. For significant changes, open an issue first to discuss.

---


## License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for the full text and terms.
