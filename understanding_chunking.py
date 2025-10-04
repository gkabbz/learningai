import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

"""
Day 2: Understanding Chunking
=============================

Step-by-step exploration of how to split text into chunks for RAG systems.
"""

# Step 1: Load and examine the transcript
def load_transcript():
    """Load the transcript file and examine its structure"""
    with open('granola_transcript.txt', 'r') as f:
        content = f.read()
    return content


def count_tokens(text):
    """Count tokens using OpenAI's tokenizer"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Load the transcript
print("📖 Loading transcript...")
transcript = load_transcript()

print(f"Total characters: {len(transcript)}")
print(f"Total words (rough): {len(transcript.split())}")

# Look at the first 500 characters to understand structure
print("\nFirst 500 characters:")
print("-" * 40)
print(transcript[:500])
print("-" * 40)

# Add after your existing code:
print(f"\nTotal tokens: {count_tokens(transcript)}")
print(f"Characters vs tokens ratio: {len(transcript) / count_tokens(transcript):.2f}")

# Check tokens in first 500 characters
first_500_chars = transcript[:500]
print(f"\nFirst 500 characters = {count_tokens(first_500_chars)} tokens")

# Step 3: Simple chunking function
def simple_chunk_by_tokens(text, chunk_size=500):
    """Split text into chunks of roughly chunk_size tokens"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(text)
    
    chunks = []
    for i in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[i:i + chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append({
            'text': chunk_text,
            'token_count': len(chunk_tokens),
            'start_token': i
        })
    
    return chunks

# Test with 100-token chunks first
print("\n🔪 Creating 100-token chunks...")
chunks_100 = simple_chunk_by_tokens(transcript, chunk_size=100)
print(f"Created {len(chunks_100)} chunks")

# Look at first few chunks
for i in range(3):
    print(f"\nChunk {i+1} ({chunks_100[i]['token_count']} tokens):")
    print(f"{chunks_100[i]['text']}")
    print("-" * 50)

 # Test with 500-token chunks
print("\n🔪 Creating 500-token chunks...")
chunks_500 = simple_chunk_by_tokens(transcript, chunk_size=500)
print(f"Created {len(chunks_500)} chunks")

# Look at first chunk
print(f"\nFirst 500-token chunk ({chunks_500[0]['token_count']} tokens):")
print(f"{chunks_500[0]['text']}")
print("-" * 50)

# Test with 1000-token chunks
print("\n🔪 Creating 1000-token chunks...")
chunks_1000 = simple_chunk_by_tokens(transcript, chunk_size=1000)
print(f"Created {len(chunks_1000)} chunks")

print(f"\nFirst 1000-token chunk ({chunks_1000[0]['token_count']} tokens):")
print(f"{chunks_1000[0]['text']}")
print("-" * 50)

# Step 4: Embed the chunks
print("\n📊 Embedding chunks with sentence-transformers...")
print("Loading model (this may take a moment first time)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Start with 500-token chunks
print(f"\nEmbedding {len(chunks_500)} chunks of 500 tokens each...")
embeddings_500 = []
for i, chunk in enumerate(chunks_500):
    embedding = model.encode(chunk['text'])
    embeddings_500.append(embedding)
    print(f"Embedded chunk {i+1}/{len(chunks_500)}", end="\r")

embeddings_500 = np.array(embeddings_500)
print(f"\n✅ Created {len(embeddings_500)} embeddings")
print(f"Each embedding has {embeddings_500[0].shape[0]} dimensions")
print(f"Total embeddings array shape: {embeddings_500.shape}")

# Step 5: Search function
def search_chunks(query, chunks, embeddings, model, top_k=3):
    """
    Search chunks for the most relevant ones to the query

    Args:
        query: The search query string
        chunks: List of chunk dictionaries
        embeddings: Numpy array of chunk embeddings
        model: The sentence transformer model
        top_k: How many top results to return

    Returns:
        List of tuples: (similarity_score, chunk_text)
    """
    # Embed the query
    query_embedding = model.encode(query)

    # Calculate similarities between query and all chunks
    # Reshape needed because cosine_similarity expects 2D arrays
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top k results
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort and reverse to get highest first

    results = []
    for idx in top_indices:
        results.append({
            'score': similarities[idx],
            'chunk_num': idx,
            'text': chunks[idx]['text']
        })

    return results

# Embed 100-token chunks
print("\n📊 Embedding 100-token chunks...")
embeddings_100 = []
for i, chunk in enumerate(chunks_100):
    embedding = model.encode(chunk['text'])
    embeddings_100.append(embedding)
    print(f"Embedded chunk {i+1}/{len(chunks_100)}", end="\r")

embeddings_100 = np.array(embeddings_100)
print(f"\n✅ Created {len(embeddings_100)} embeddings (100-token chunks)")

# Embed 1000-token chunks
print("\n📊 Embedding 1000-token chunks...")
embeddings_1000 = []
for i, chunk in enumerate(chunks_1000):
    embedding = model.encode(chunk['text'])
    embeddings_1000.append(embedding)
    print(f"Embedded chunk {i+1}/{len(chunks_1000)}", end="\r")

embeddings_1000 = np.array(embeddings_1000)
print(f"\n✅ Created {len(embeddings_1000)} embeddings (1000-token chunks)")

# Step 6: Compare retrieval across all chunk sizes
print("\n" + "=" * 80)
print("COMPARING RETRIEVAL QUALITY ACROSS CHUNK SIZES")
print("=" * 80)

test_query = "Who is Toby?"
print(f"\nQuery: '{test_query}'\n")

# Test 100-token chunks
print("\n--- 100-TOKEN CHUNKS ---")
results_100 = search_chunks(test_query, chunks_100, embeddings_100, model, top_k=1)
print(f"Top result - Chunk {results_100[0]['chunk_num']} (similarity: {results_100[0]['score']:.3f})")
print(f"Text: {results_100[0]['text'][:300]}...")

# Test 500-token chunks
print("\n--- 500-TOKEN CHUNKS ---")
results_500 = search_chunks(test_query, chunks_500, embeddings_500, model, top_k=1)
print(f"Top result - Chunk {results_500[0]['chunk_num']} (similarity: {results_500[0]['score']:.3f})")
print(f"Text: {results_500[0]['text'][:300]}...")

# Test 1000-token chunks
print("\n--- 1000-TOKEN CHUNKS ---")
results_1000 = search_chunks(test_query, chunks_1000, embeddings_1000, model, top_k=1)
print(f"Top result - Chunk {results_1000[0]['chunk_num']} (similarity: {results_1000[0]['score']:.3f})")
print(f"Text: {results_1000[0]['text'][:300]}...")

# Summary comparison
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"100-token chunks:  {results_100[0]['score']:.3f} similarity")
print(f"500-token chunks:  {results_500[0]['score']:.3f} similarity")
print(f"1000-token chunks: {results_1000[0]['score']:.3f} similarity")