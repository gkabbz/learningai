"""
Day 9: Firestore Vector Search Basics
Goal: Store embeddings in Firestore and prepare for similarity search
"""

import os
from dotenv import load_dotenv
from google.cloud import firestore
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded (384 dimensions)\n")

print("Connecting to Firestore...")
db = firestore.Client(project=os.getenv('GCP_PROJECT'))
print("✓ Connected!\n")

# Fix: Re-store embeddings as Vector type (not arrays!)
from google.cloud.firestore_v1.vector import Vector

print("Fixing embeddings - converting from arrays to Vector type...")

# Fix meeting_001 chunk
chunk1_ref = db.collection('meetings').document('meeting_001').collection('chunks').document('chunk_001')
chunk1 = chunk1_ref.get()
if chunk1.exists:
    text1 = chunk1.to_dict()['text']
    embedding1 = model.encode(text1)
    chunk1_ref.update({'embedding': Vector(embedding1.tolist())})
    print(f"✓ Fixed: {chunk1_ref.path}")

# Fix meeting_002 chunk
chunk2_ref = db.collection('meetings').document('meeting_002').collection('chunks').document('chunk_001')
chunk2 = chunk2_ref.get()
if chunk2.exists:
    text2 = chunk2.to_dict()['text']
    embedding2 = model.encode(text2)
    chunk2_ref.update({'embedding': Vector(embedding2.tolist())})
    print(f"✓ Fixed: {chunk2_ref.path}")

print("\nUsing existing chunk in Firestore: meetings/meeting_002/chunks/chunk_001")

# Now let's search for similar chunks
print("\n" + "="*60)
print("SIMILARITY SEARCH")
print("="*60)

query_text = "Who talked about battleships?"
print(f"\nQuery: '{query_text}'")
print("Generating query embedding...")
query_embedding = model.encode(query_text)
print(f"✓ Query embedding generated ({len(query_embedding)} dimensions)")

# First, let's verify the chunk exists
print("\nVerifying chunk exists...")
test_chunk = db.collection('meetings').document('meeting_002').collection('chunks').document('chunk_001').get()
if test_chunk.exists:
    print(f"✓ Chunk found: {test_chunk.to_dict()['text'][:50]}...")
    print(f"  Has embedding: {'embedding' in test_chunk.to_dict()}")
    print(f"  Embedding length: {len(test_chunk.to_dict()['embedding'])}")
else:
    print("✗ Chunk not found!")

# Search Firestore for similar chunks using vector search
print("\nSearching Firestore for similar chunks...")
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

# First try: list all chunks to see if collection_group works
print("Debug: Listing all chunks in collection group...")
all_chunks = db.collection_group('chunks').stream()
all_chunks_list = list(all_chunks)
print(f"  Total chunks found via collection_group: {len(all_chunks_list)}")
for chunk in all_chunks_list:
    print(f"    - {chunk.reference.path}")

# Query all chunks collections across all meetings
print("\nNow trying vector search...")
chunks_collection = db.collection_group('chunks')
vector_query = chunks_collection.find_nearest(
    vector_field='embedding',
    query_vector=Vector(query_embedding.tolist()),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=3
)

results = vector_query.stream()
results_list = list(results)
print(f"✓ Found {len(results_list)} results\n")

for i, doc in enumerate(results_list):
    data = doc.to_dict()
    print(f"Result {i+1}:")
    print(f"  Text: {data['text']}")
    print(f"  Path: {doc.reference.path}")
    print()
