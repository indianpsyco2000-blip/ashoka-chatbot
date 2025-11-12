from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # pip install openai

app = Flask(__name__)
CORS(app)

# Grok API Setup
GROK_API_KEY = os.getenv('GROK_API_KEY')
if not GROK_API_KEY:
    raise ValueError("Set GROK_API_KEY env var")
client = OpenAI(api_key=GROK_API_KEY, base_url="https://api.x.ai/v1")

# Load & Embed Data
print("Loading ashoka_info.txt...")
with open('ashoka_info.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks

chunks = chunk_text(text)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(chunks, show_progress_bar=False).astype('float32')
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)
print(f"Indexed {len(chunks)} chunks. Ready!")

def retrieve_context(query, k=3):
    q_vec = embedder.encode([query]).astype('float32')
    _, I = faiss_index.search(q_vec, k)
    return "\n\n".join([chunks[i] for i in I[0]])

PROMPT = """
You are an AI assistant for Ashoka Institute of Technology and Management, Varanasi.
Answer ONLY using the context. If unrelated, say: "I can only help with Ashoka Institute information."

Context: {context}

Question: {query}
Answer:
"""

@app.route('/')
def index():
    with open('chatbot.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message', '').strip()
    if not user_msg:
        return jsonify({'response': 'Please type a question.'})

    context = retrieve_context(user_msg, k=3)
    full_prompt = PROMPT.format(context=context, query=user_msg)

    try:
        response = client.chat.completions.create(
            model="grok-beta",
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=500,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"API error: {str(e)}. Check key."

    return jsonify({'response': answer})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)