from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# === OPENROUTER (FREE) ===
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# === LOAD DATA ===
try:
    with open('ashoka_info.txt', 'r', encoding='utf-8') as f:
        text = f.read()
except:
    text = "Ashoka Institute offers B.Tech, M.Tech, MBA. Contact: info@ashoka.edu.in"

def chunk(text, size=500, overlap=50):
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

chunks = chunk(text)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embedder.encode(chunks).astype('float32')
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)
print(f"Indexed {len(chunks)} chunks.")

def retrieve(query, k=3):
    q = embedder.encode([query]).astype('float32')
    _, I = index.search(q, k)
    return "\n\n".join([chunks[i] for i in I[0]])

PROMPT = """
You are AI assistant for Ashoka Institute of Technology and Management, Varanasi.
Answer ONLY using context. If unrelated: "I can only help with institute info."

Context: {context}
Question: {query}
Answer:
"""

@app.route('/')
def home():
    return open('chatbot.html', 'r', encoding='utf-8').read()

@app.route('/chat', methods=['POST'])
def chat():
    msg = request.json.get('message', '').strip()
    if not msg:
        return jsonify({'response': 'Ask something!'})

    context = retrieve(msg)
    prompt = PROMPT.format(context=context, query=msg)

    try:
        resp = client.chat.completions.create(
            model="deepseek/deepseek-r1:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        answer = f"AI error: {e}"

    return jsonify({'response': answer})

# Vercel entry point
def handler(event, context=None):
    from werkzeug.serving import run_simple
    return run_simple('0.0.0.0', int(os.environ.get('PORT', 3000)), app)