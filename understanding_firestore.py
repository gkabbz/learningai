"""
Day 8: Understanding Firestore Fundamentals
Goal: Connect to Firestore and create a simple document
"""

import os
from dotenv import load_dotenv
from google.cloud import firestore

# Load environment variables
load_dotenv()

print("Connecting to Firestore...")
db = firestore.Client(project=os.getenv('GCP_PROJECT'))
print("✓ Connected!\n")

# Create a simple meeting document
meeting_data = {
    'title': 'West Wing Strategy Session',
    'date': '2024-10-01',
    'participants': ['Toby', 'Sam', 'Josh']
}

print("Creating meeting document...")
db.collection('meetings').document('meeting_001').set(meeting_data)
print("✓ Document created!\n")

print("Meeting stored:")
print(f"  Path: meetings/meeting_001")
print(f"  Data: {meeting_data}")

# Read it back to verify
print("\nReading document back from Firestore...")
doc_ref = db.collection('meetings').document('meeting_001')
doc = doc_ref.get()

if doc.exists:
    print("✓ Document found!")
    retrieved_data = doc.to_dict()
    print(f"  Retrieved: {retrieved_data}")
    print(f"  Participants: {retrieved_data['participants']}")
else:
    print("✗ Document not found")

# Now add a chunk as a SUBCOLLECTION
print("\nAdding a chunk as subcollection...")
chunk_data = {
    'text': 'Toby discusses the battleship with Sam and Josh.',
    'chunk_index': 0
}

# Path: meetings/meeting_001/chunks/chunk_001
chunk_ref = db.collection('meetings').document('meeting_001').collection('chunks').document('chunk_001')
chunk_ref.set(chunk_data)
print("✓ Chunk added as subcollection!")
print(f"  Path: meetings/meeting_001/chunks/chunk_001")

# Read the chunk back
print("\nReading chunk back from subcollection...")
retrieved_chunk_ref = db.collection('meetings').document('meeting_001').collection('chunks').document('chunk_001')
chunk_doc = retrieved_chunk_ref.get()

if chunk_doc.exists:
    print("✓ Chunk found!")
    chunk_retrieved = chunk_doc.to_dict()
    print(f"  Text: {chunk_retrieved['text']}")
else:
    print("✗ Chunk not found")
