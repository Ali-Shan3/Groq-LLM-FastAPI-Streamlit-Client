# Groq-LLM-FastAPI-Streamlit-Client
A complete backend + frontend demo using Groq (OpenAI-compatible) API, FastAPI, and Streamlit.
This project provides:

✅ A FastAPI backend with:
Text generation endpoint
Chat endpoint
Image + instruction (Vision) endpoint
CRUD API
Groq Responses API integration

✅ A Streamlit client with:
Text Chat UI
Image Captioning / Q&A UI
CRUD UI for demo items

📂 Project Structure
.
├── main.py                # FastAPI backend
├── streamlit_client.py    # Streamlit UI client
├── .env                   # API keys (ignored in Git)
├── requirements.txt
└── README.md


⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO


🧪 Backend Setup (FastAPI)

2️⃣ Create Virtual Environment
python -m venv env
Activate it:
Windows:
env\Scripts\activate

Mac/Linux:
source env/bin/activate


3️⃣ Install Dependencies
pip install -r requirements.txt

If you don't have a requirements.txt, use:
pip install fastapi uvicorn python-dotenv httpx pillow streamlit requests


4️⃣ Add your Groq API Key
Create a .env file:
GROQ_API_KEY=your_api_key_here
DEFAULT_GROQ_MODEL=openai/gpt-oss-20b

Or edit in main.py:
GROQ_API_KEY = "Enter_Your_API_Here"


5️⃣ Run FastAPI Server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

FastAPI Swagger docs available at:
👉 http://localhost:8000/docs
👉 http://localhost:8000/redoc


🎨 Frontend Setup (Streamlit Client)
Run the Streamlit UI:
streamlit run streamlit_client.py

You will see:
Backend URL field → default: http://localhost:8000
Tabs:
    Text Chat
    Image Description (Vision)
    CRUD Items Manager


🧠 API Endpoints Summary
🔹 Text LLM
POST /llm/text

Body:
{
  "prompt": "Hello!",
  "model": "openai/gpt-oss-20b"
}


🔹 Chat LLM
POST /llm/chat

Body example:
{
  "messages": [
    { "role": "user", "content": "Tell me a joke." }
  ]
}


🔹 Image + Instruction (Vision LLM)
POST /llm/image

Form-data:
    file: image
    instruction: text
    model: optional


🔹 CRUD Endpoints
    Method          Route               Description
    GET             /items              List items
    GET             /items/{id}         Get item by ID
    POST            /items              Create item
    PUT             /items/{id}         Update item
    DELETE          /items/{id}         Delete item


🛠 Example FastAPI Run
    uvicorn main:app --reload

Output:
    Uvicorn running at http://127.0.0.1:8000
