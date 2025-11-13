from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

app = Flask(__name__, static_folder='../static')
CORS(app)

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# FIXED: File is now in same folder
with open('ashoka_info.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def chunk(text, size=500, overlap=50):
    return [text[i:i+size] for i in range(0, len(text), size-overlap)]

chunks = chunk(text)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
vectors = embedder.encode(chunks).astype('float32')
index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

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
    return send_from_directory('../static', 'chatbot.html')

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
        answer = "Sorry, AI busy. Try again."

    return jsonify({'response': answer})

if __name__ == '__main__':
    app.run()