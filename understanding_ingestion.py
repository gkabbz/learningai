import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
import tiktoken

"""
Day 12: Ingestion Pipeline + Duplicate Detection
=================================================

Build production-ready ingestion that:
- Checks if PR already exists
- Skips duplicates
- Only processes new PRs
"""

# Load environment
load_dotenv()

# Initialize
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✓ Model loaded!")

print("Connecting to Firestore...")
db = firestore.Client(project=os.getenv('GCP_PROJECT'))
print("✓ Firestore connected!")


def pr_exists(pr_number, db):
    """
    Check if a PR already exists in Firestore

    Args:
        pr_number: The PR number (e.g., 7974)
        db: Firestore client

    Returns:
        True if PR exists, False otherwise
    """
    pr_ref = db.collection('prs').document(f'pr_{pr_number}')
    doc = pr_ref.get()

    return doc.exists


def count_tokens(text):
    """Count tokens using OpenAI's tokenizer"""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))


def chunk_pr_overview(pr_data):
    """Create the overview chunk for a PR"""
    reviewers = list(set([review['user']['login'] for review in pr_data['reviews']]))

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
    """Create the files chunk for a PR"""
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
    """Create review chunks for a PR"""
    review_chunks = []

    for review in pr_data['reviews']:
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
    """Chunk a complete PR into multiple searchable pieces"""
    chunks = []
    chunks.append(chunk_pr_overview(pr_data))
    chunks.append(chunk_pr_files(pr_data))
    chunks.extend(chunk_pr_reviews(pr_data))
    return chunks


def generate_embeddings(chunks, model):
    """Generate embeddings for all chunks"""
    for chunk in chunks:
        embedding = model.encode(chunk['text'])
        chunk['embedding'] = embedding
    return chunks


def store_pr_in_firestore(pr_data, chunks, db):
    """Store a PR and its chunks in Firestore"""
    pr_number = pr_data['number']

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

    # Store each chunk as a subcollection
    for i, chunk in enumerate(chunks):
        chunk_ref = pr_ref.collection('chunks').document(f'chunk_{i+1}')
        chunk_ref.set({
            'type': chunk['type'],
            'text': chunk['text'],
            'token_count': chunk['token_count'],
            'embedding': Vector(chunk['embedding'].tolist())
        })


def ingest_pr_with_duplicate_check(pr_data, db, model):
    """
    Ingest a PR into Firestore with duplicate detection

    Args:
        pr_data: The PR data loaded from JSON
        db: Firestore client
        model: Sentence transformer model

    Returns:
        'skipped' if duplicate, 'ingested' if newly added
    """
    pr_number = pr_data['number']

    # Check if PR already exists
    if pr_exists(pr_number, db):
        print(f"  ⏭️  PR #{pr_number} skipped (already loaded)")
        return 'skipped'

    # PR doesn't exist, process it
    print(f"  📝 PR #{pr_number} is NEW, processing...")

    # Create chunks
    chunks = chunk_pr(pr_data)
    print(f"     Created {len(chunks)} chunks")

    # Generate embeddings
    chunks = generate_embeddings(chunks, model)
    print(f"     Generated embeddings")

    # Store in Firestore
    store_pr_in_firestore(pr_data, chunks, db)
    print(f"  ✅ PR #{pr_number} ingested successfully!")

    return 'ingested'


# Test the function
if __name__ == "__main__":
    import glob

    print("\n" + "="*60)
    print("TESTING SMART INGESTION WITH DUPLICATE DETECTION")
    print("="*60)

    # Find PR files
    pr_cache_dir = '/Users/gkabbz/PycharmProjects/github-delivery-visibility/cache/mozilla_bigquery-etl'
    pr_files = glob.glob(f'{pr_cache_dir}/pr_*.json')

    print(f"\nFound {len(pr_files)} PR files in cache")
    print(f"Processing ALL PRs...\n")

    # Process all PRs
    stats = {'skipped': 0, 'ingested': 0}

    for pr_path in pr_files:
        # Load PR data from JSON
        with open(pr_path, 'r') as f:
            pr_data = json.load(f)

        pr_number = pr_data['number']
        print(f"Processing PR #{pr_number}: {pr_data['title'][:60]}...")

        # Ingest with duplicate check
        result = ingest_pr_with_duplicate_check(pr_data, db, model)
        stats[result] += 1
        print()

    # Show summary
    print("="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Total PRs processed: {stats['skipped'] + stats['ingested']}")
    print(f"✅ Newly ingested: {stats['ingested']}")
    print(f"⏭️  Skipped (duplicates): {stats['skipped']}")
    print()
    print("💡 Key learning: Duplicate detection prevents re-processing!")
    print("   Run this script again and all 5 should be skipped.")
