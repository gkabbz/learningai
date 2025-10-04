import chromadb
import tiktoken

"""
Day 3: Understanding ChromaDB
==============================

Learning how to persist embeddings and build a queryable vector database.
"""

# Helper function from Day 2
def load_transcript():
    """Load the transcript file"""
    with open('granola_transcript.txt', 'r') as f:
        content = f.read()
    return content

def simple_chunk_by_tokens(text, chunk_size=1000):
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

# Step 1: Load and chunk the transcript
print("📖 Loading transcript...")
transcript = load_transcript()

print("🔪 Creating 1000-token chunks (our optimal size from Day 2)...")
chunks = simple_chunk_by_tokens(transcript, chunk_size=1000)
print(f"✅ Created {len(chunks)} chunks")

# Step 2: Create a ChromaDB client
print("\n📦 Creating ChromaDB client...")
client = chromadb.Client()

# Step 3: Create a collection
print("📚 Creating a collection called 'meeting_transcripts'...")
collection = client.create_collection(name="meeting_transcripts")

print(f"✅ Collection created: {collection.name}")
print(f"   Items in collection: {collection.count()}")

# Step 4: Add chunks to the collection
print("\n💾 Adding chunks to ChromaDB...")

# Prepare the data
documents = [chunk['text'] for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Add to collection (ChromaDB will auto-generate embeddings!)
collection.add(
    documents=documents,
    ids=ids
)

print(f"✅ Added {len(documents)} chunks to collection")
print(f"   Items in collection now: {collection.count()}")

# Step 5: Query the collection
print("\n🔍 Testing multiple queries...")

test_queries = [
    "Who is Toby?",
    "What was discussed about the president?",
    "Tell me about military or battleships"
]

for query in test_queries:
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print('='*60)

    results = collection.query(
        query_texts=[query],
        n_results=2  # Get top 2 results
    )

    for i in range(len(results['ids'][0])):
        print(f"\nResult {i+1}:")
        print(f"  ID: {results['ids'][0][i]}")
        print(f"  Distance: {results['distances'][0][i]:.3f}")
        print(f"  Text: {results['documents'][0][i][:250]}...")
