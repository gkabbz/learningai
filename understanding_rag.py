from dotenv import load_dotenv
import os
import anthropic
import chromadb
import tiktoken

"""
Day 5: Understanding RAG (Retrieval Augmented Generation)
==========================================================

Connecting retrieval (ChromaDB) with generation (Claude API) to answer questions
based on your personal knowledge base.
"""

# Load environment variables and create Claude client
load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

print("✅ Claude API client ready")

# Helper functions
def load_transcript():
    with open('granola_transcript.txt', 'r') as f:
        return f.read()

def simple_chunk_by_tokens(text, chunk_size=1000):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({'text': chunk_text})
    return chunks

# Set up ChromaDB with transcript
print("\n📖 Loading transcript and setting up ChromaDB...")
transcript = load_transcript()
chunks = simple_chunk_by_tokens(transcript, chunk_size=1000)

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="rag_demo")

documents = [chunk['text'] for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(documents=documents, ids=ids)
print(f"✅ Added {len(chunks)} chunks to ChromaDB")

# Step 1: Test Claude WITHOUT context (baseline)
print("\n" + "="*60)
print("TEST 1: Claude WITHOUT context")
print("="*60)

question = "Who is Toby?"
print(f"\nQuestion: {question}")

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[{"role": "user",
        "content": question
    }]
)

print(f"\nClaude's answer (no context):")
print(response.content[0].text)

# Step 2: Retrieve relevant chunks from ChromaDB
print("\n" + "="*60)
print("TEST 2: Claude WITH context (RAG)")
print("="*60)

print(f"\nSame question: {question}")
print("\nRetrieving relevant chunks from ChromaDB...")

results = collection.query(
    query_texts=[question],
    n_results=2  # Get top 2 most relevant chunks
)

print(f"Found {len(results['documents'][0])} relevant chunks")

# Build the context from retrieved chunks
context = "\n\n".join(results['documents'][0])
print(f"\nContext length: {len(context)} characters")

# Create RAG prompt with context
rag_prompt = f"""You are a helpful assistant that answers questions based on provided context.

Context from meeting transcripts:
{context}

Question: {question}

Please answer the question based ONLY on the information in the context above. If the context doesn't contain enough information to answer the question, say so."""

print("\nSending context + question to Claude...")

rag_response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[{
        "role": "user",
        "content": rag_prompt
    }]
)

print(f"\nClaude's answer (WITH context - RAG):")
print(rag_response.content[0].text)

# Step 3: Follow-up question with conversation history
print("\n" + "="*60)
print("TEST 3: Follow-up question (multi-turn conversation)")
print("="*60)

follow_up_question = "What else can you tell me about him?"
print(f"\nFollow-up: {follow_up_question}")

follow_up_response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=200,
    messages=[
        {"role": "user", "content": rag_prompt},
        {"role": "assistant", "content": rag_response.content[0].text},
        {"role": "user", "content": follow_up_question}
    ]
)

print(f"\nClaude's follow-up answer:")
print(follow_up_response.content[0].text)