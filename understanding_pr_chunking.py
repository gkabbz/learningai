import json
import os
import tiktoken
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector

"""
Day 10: Understanding PR Chunking for Firestore
================================================

Chunk PR data from GitHub into searchable pieces for vector storage
"""

# Load environment variables
load_dotenv()

# Load the embedding model (same as Week 1)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# Connect to Firestore
print("Connecting to Firestore...")
db = firestore.Client(project=os.getenv('GCP_PROJECT'))
print("Connected to Firestore!")

def count_tokens(text):
    """Count tokens using OpenAI's tokenizer"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def chunk_pr_overview(pr_data):
    """
    Create the overview chunk for a PR

    Contains:
    - PR number, title, state
    - Author
    - Created/merged dates
    - Body (description)
    - List of reviewers
    """
    # Get unique reviewers
    reviewers = list(set([review['user']['login'] for review in pr_data['reviews']]))

    # Build the overview text
    overview_text = f"""Pull Request #{pr_data['number']}: {pr_data['title']}

Status: {pr_data['state']}
Author: {pr_data['author']['login']}
Created: {pr_data['created_at']}
Merged: {pr_data.get('merged_at', 'Not merged')}

Description:
{pr_data['body']}

Reviewers: {', '.join(sorted(reviewers))}
""".strip()

    return {
        'type': 'overview',
        'text': overview_text,
        'token_count': count_tokens(overview_text)
    }


def chunk_pr_files(pr_data):
    """
    Create the files chunk for a PR

    Contains:
    - List of all files changed
    - Status (added/modified/removed)
    - Number of additions/deletions
    """
    files_list = []
    for file_stat in pr_data['file_stats']:
        files_list.append(
            f"- {file_stat['filename']} ({file_stat['status']}): "
            f"+{file_stat['additions']} -{file_stat['deletions']}"
        )

    files_text = f"""Files Changed in PR #{pr_data['number']}:

{chr(10).join(files_list)}

Total files changed: {len(pr_data['file_stats'])}""".strip()

    return {
        'type': 'files',
        'text': files_text,
        'token_count': count_tokens(files_text)
    }


def chunk_pr_reviews(pr_data):
    """
    Create review chunks for a PR

    Only creates chunks for reviews that have a body (actual comments)
    Each review with content becomes its own chunk

    Returns a list of review chunks
    """
    review_chunks = []

    for review in pr_data['reviews']:
        # Only create chunks for reviews with actual text
        if review['body'] and review['body'].strip():
            review_text = f"""Review by {review['user']['login']} on PR #{pr_data['number']}
State: {review['state']}
Submitted: {review['submitted_at']}

Comment:
{review['body']}""".strip()

            review_chunks.append({
                'type': 'review',
                'text': review_text,
                'token_count': count_tokens(review_text)
            })

    return review_chunks


def chunk_pr(pr_data):
    """
    Chunk a complete PR into multiple searchable pieces

    Returns a list of chunks with text and token counts
    """
    chunks = []

    # Chunk 1: Overview
    chunks.append(chunk_pr_overview(pr_data))

    # Chunk 2: Files changed
    chunks.append(chunk_pr_files(pr_data))

    # Chunk 3+: Reviews with comments
    review_chunks = chunk_pr_reviews(pr_data)
    chunks.extend(review_chunks)

    return chunks


def generate_embeddings(chunks, model):
    """
    Generate embeddings for all chunks

    Adds 'embedding' field to each chunk dictionary
    """
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks, 1):
        embedding = model.encode(chunk['text'])
        chunk['embedding'] = embedding
        print(f"  {i}/{len(chunks)}: {chunk['type']} ({len(embedding)} dimensions)")

    print("Embeddings complete!")
    return chunks


def store_pr_in_firestore(pr_data, chunks, db):
    """
    Store a PR and its chunks in Firestore

    Structure: prs/{pr_number}/chunks/{chunk_id}
    """
    pr_number = pr_data['number']
    print(f"\nStoring PR #{pr_number} in Firestore...")

    # Create the PR document
    pr_ref = db.collection('prs').document(f'pr_{pr_number}')
    pr_ref.set({
        'number': pr_number,
        'title': pr_data['title'],
        'author': pr_data['author']['login'],
        'state': pr_data['state'],
        'created_at': pr_data['created_at'],
        'merged_at': pr_data.get('merged_at')
    })
    print(f"  ✓ Created PR document: {pr_ref.path}")

    # Store each chunk as a subcollection
    for i, chunk in enumerate(chunks):
        chunk_ref = pr_ref.collection('chunks').document(f'chunk_{i+1}')

        # Convert embedding to Vector type (required for Firestore vector search)
        chunk_ref.set({
            'type': chunk['type'],
            'text': chunk['text'],
            'token_count': chunk['token_count'],
            'embedding': Vector(chunk['embedding'].tolist())
        })
        print(f"  ✓ Stored chunk {i+1}/{len(chunks)}: {chunk['type']}")

    print(f"✓ PR #{pr_number} stored successfully!")


def search_chunks(query, db, model, limit=3):
    """
    Search for chunks similar to a query using vector search

    Args:
        query: The search query string
        db: Firestore client
        model: Sentence transformer model
        limit: Number of results to return
    """
    print(f"\n{'='*60}")
    print(f"VECTOR SEARCH")
    print(f"{'='*60}")
    print(f"Query: '{query}'")

    # Generate embedding for the query
    query_embedding = model.encode(query)
    print(f"Query embedding: {len(query_embedding)} dimensions")

    # Search using Firestore vector search
    print(f"\nSearching for {limit} most similar chunks...")

    # Get all chunk references (from all PRs)
    chunks_query = db.collection_group('chunks').where(
        filter=firestore.FieldFilter('embedding', '!=', None)
    )

    # Use find_nearest for vector search
    vector_query = chunks_query.find_nearest(
        vector_field='embedding',
        query_vector=Vector(query_embedding.tolist()),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=limit
    )

    results = vector_query.stream()

    print("\nResults:")
    for i, doc in enumerate(results, 1):
        data = doc.to_dict()
        pr_id = doc.reference.parent.parent.id  # Get parent PR document ID
        print(f"\n{i}. {pr_id} - {data['type']}")
        print(f"   Text preview: {data['text'][:150]}...")

    return results


# Process all PRs
if __name__ == "__main__":
    import glob

    # Find all PR files
    pr_cache_dir = '/Users/gkabbz/PycharmProjects/github-delivery-visibility/cache/mozilla_bigquery-etl'
    pr_files = glob.glob(f'{pr_cache_dir}/pr_*.json')

    print(f"\nFound {len(pr_files)} PR files")
    print("Starting batch processing...\n")

    # Process each PR
    for i, pr_path in enumerate(pr_files, 1):
        with open(pr_path, 'r') as f:
            pr = json.load(f)

        print(f"[{i}/{len(pr_files)}] PR #{pr['number']}: {pr['title'][:60]}...")

        # Create chunks
        chunks = chunk_pr(pr)
        print(f"  Created {len(chunks)} chunks")

        # Generate embeddings
        chunks = generate_embeddings(chunks, model)

        # Store in Firestore
        store_pr_in_firestore(pr, chunks, db)

        print(f"  ✓ Complete!\n")

    print(f"🎉 All {len(pr_files)} PRs loaded!")

    # Test vector search
    print("\n" + "="*60)
    print("Testing vector search across all PRs...")
    print("="*60)
    search_chunks("Who reviewed PRs about baseline tables?", db, model)