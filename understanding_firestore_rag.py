import os
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

"""
Day 11: RAG Query Pipeline with Firestore
==========================================

Connect vector search with Claude API for natural language answers
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

print("Connecting to Claude API...")
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
print("✓ Claude API connected!")


def retrieve_chunks(query, limit=5, date_range=None, author=None, reviewer=None):
    """
    Retrieve relevant chunks using vector search with optional filters

    Args:
        query: The user's question
        limit: Number of chunks to retrieve
        date_range: Tuple of (start_date, end_date) strings (e.g., ("2025-10-01", "2025-10-15"))
        author: Filter by PR author username
        reviewer: Filter by reviewer username

    Returns:
        List of chunk documents
    """
    print(f"\nRetrieving chunks for: '{query}'")
    if date_range:
        print(f"  Date filter: {date_range[0]} to {date_range[1]}")
    if author:
        print(f"  Author filter: {author}")
    if reviewer:
        print(f"  Reviewer filter: {reviewer}")

    # Step 1: If filters provided, get filtered PR IDs first
    filtered_pr_ids = None

    if date_range or author:
        print("\n  Applying PR-level filters...")
        prs_query = db.collection('prs')

        # Add date filter
        if date_range:
            start_date, end_date = date_range
            prs_query = prs_query.where('merged_at', '>=', start_date)
            prs_query = prs_query.where('merged_at', '<=', end_date)

        # Add author filter
        if author:
            prs_query = prs_query.where('author', '==', author)

        # Get filtered PRs
        filtered_prs = prs_query.stream()
        filtered_pr_ids = [pr.id for pr in filtered_prs]
        print(f"  Found {len(filtered_pr_ids)} PRs matching filters")

        if len(filtered_pr_ids) == 0:
            print("  No PRs match the filters!")
            return []

    # Step 2: Generate embedding for the query
    query_embedding = model.encode(query)
    print(f"  Generated query embedding: {len(query_embedding)} dimensions")

    # Step 3: Vector search (either across all chunks or filtered chunks)
    chunks_query = db.collection_group('chunks').where(
        filter=firestore.FieldFilter('embedding', '!=', None)
    )

    vector_query = chunks_query.find_nearest(
        vector_field='embedding',
        query_vector=Vector(query_embedding.tolist()),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=limit * 3 if filtered_pr_ids else limit  # Get more, then filter
    )

    results = vector_query.stream()

    # Step 4: If we have PR filters, only keep chunks from those PRs
    if filtered_pr_ids:
        filtered_results = []
        for chunk in results:
            pr_id = chunk.reference.parent.parent.id
            if pr_id in filtered_pr_ids:
                filtered_results.append(chunk)
                if len(filtered_results) >= limit:
                    break
        return filtered_results

    return list(results)


def build_context(chunks):
    """
    Build context prompt from retrieved chunks with PR metadata

    Args:
        chunks: List of chunk documents from Firestore

    Returns:
        Formatted context string for Claude
    """
    context_parts = []

    for i, chunk_doc in enumerate(chunks, 1):
        chunk_data = chunk_doc.to_dict()

        # Get parent PR document
        pr_ref = chunk_doc.reference.parent.parent
        pr_doc = pr_ref.get()
        pr_data = pr_doc.to_dict()

        # Format this chunk with metadata
        source = f"""Source {i}:
- PR #{pr_data['number']}: {pr_data['title']}
- Author: {pr_data['author']}
- State: {pr_data['state']}
- Merged: {pr_data.get('merged_at', 'Not merged')}
- Chunk Type: {chunk_data['type']}

Content:
{chunk_data['text']}
"""
        context_parts.append(source)

    return "\n---\n".join(context_parts)


def ask_claude(query, context):
    """
    Send query + context to Claude for answer generation

    Args:
        query: The user's question
        context: Formatted context from retrieved chunks

    Returns:
        Claude's answer
    """
    prompt = f"""You are a helpful assistant that answers questions about GitHub pull requests based on provided context.

Context from PR database:
{context}

Question: {query}

Please answer based ONLY on the information in the sources above.
If the sources don't contain enough information to answer, say so.
At the end of your answer, cite which source(s) you used (e.g., "Source: PR #8170, PR #8212").
"""

    print("\nSending to Claude API...")

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text


# Test end-to-end RAG pipeline
if __name__ == "__main__":
    question = "What did benwu review last 2 weeks?"

    print("\n" + "=" * 60)
    print("END-TO-END RAG TEST WITH DATE FILTER")
    print("=" * 60)
    print(f"Question: {question}\n")

    # Step 1: Retrieve relevant chunks with date filter
    chunks = retrieve_chunks(
        question,
        limit=5,
        date_range=("2025-10-01", "2025-10-15")
    )
    print(f"✓ Retrieved {len(chunks)} chunks")

    # Step 2: Build context with metadata
    context = build_context(chunks)
    print(f"✓ Built context with PR metadata")

    # Step 3: Get answer from Claude
    answer = ask_claude(question, context)

    # Display answer
    print("\n" + "=" * 60)
    print("CLAUDE'S ANSWER:")
    print("=" * 60)
    print(answer)
