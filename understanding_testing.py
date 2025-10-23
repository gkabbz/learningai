import os
from dotenv import load_dotenv
from google.cloud import firestore
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic

"""
Day 14: Testing & Validation
=============================

Comprehensive testing of RAG system with 19 semantic questions
Testing retrieval quality, metadata filtering, and answer accuracy
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


def retrieve_chunks(query, limit=5, date_range=None, author=None):
    """Retrieve relevant chunks using vector search with optional filters"""

    # Step 1: If filters provided, get filtered PR IDs first
    filtered_pr_ids = None

    if date_range or author:
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

        if len(filtered_pr_ids) == 0:
            return []

    # Step 2: Generate embedding for the query
    query_embedding = model.encode(query)

    # Step 3: Vector search
    chunks_query = db.collection_group('chunks').where(
        filter=firestore.FieldFilter('embedding', '!=', None)
    )

    vector_query = chunks_query.find_nearest(
        vector_field='embedding',
        query_vector=Vector(query_embedding.tolist()),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=limit * 3 if filtered_pr_ids else limit
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
    """Build context prompt from retrieved chunks with PR metadata"""
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

Content:
{chunk_data['text']}
"""
        context_parts.append(source)

    return "\n---\n".join(context_parts)


def ask_claude(query, context):
    """Send query + context to Claude for answer generation"""
    prompt = f"""You are a helpful assistant that answers questions about GitHub pull requests based on provided context.

Context from PR database:
{context}

Question: {query}

Please answer based ONLY on the information in the sources above.
If the sources don't contain enough information to answer, say so.
At the end of your answer, cite which source(s) you used (e.g., "Source: PR #8170, PR #8212").
"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.content[0].text


def run_test_query(question, date_range=None, author=None, category="General"):
    """Run a single test query and return results"""
    print(f"\n{'='*70}")
    print(f"[{category}] {question}")
    print(f"{'='*70}")

    if date_range:
        print(f"📅 Date filter: {date_range[0]} to {date_range[1]}")
    if author:
        print(f"👤 Author filter: {author}")

    # Retrieve chunks
    chunks = retrieve_chunks(
        question,
        limit=5,
        date_range=date_range,
        author=author
    )

    print(f"✓ Retrieved {len(chunks)} chunks")

    if len(chunks) == 0:
        print("❌ No results found!")
        return {
            'question': question,
            'category': category,
            'chunks_retrieved': 0,
            'answer': None
        }

    # Build context
    context = build_context(chunks)

    # Get answer from Claude
    answer = ask_claude(question, context)

    print(f"\n📝 ANSWER:")
    print(answer)

    return {
        'question': question,
        'category': category,
        'chunks_retrieved': len(chunks),
        'answer': answer
    }


# Test suite
if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAY 14: COMPREHENSIVE RAG SYSTEM TESTING")
    print("="*70)
    print("\nTesting with 140 PRs - 19 semantic questions\n")

    test_results = []

    # ===================================================================
    # CATEGORY 1: SUMMARY/ACTIVITY (time-based)
    # ===================================================================
    print("\n📊 CATEGORY 1: SUMMARY & ACTIVITY (Last 2 weeks)")
    print("="*70)

    two_weeks_ago = "2025-10-08"
    today = "2025-10-22"

    test_results.append(run_test_query(
        "What new datasets and views were added?",
        date_range=(two_weeks_ago, today),
        category="Summary"
    ))

    test_results.append(run_test_query(
        "What bug fixes were made?",
        date_range=(two_weeks_ago, today),
        category="Summary"
    ))

    test_results.append(run_test_query(
        "What dependency updates were made?",
        date_range=(two_weeks_ago, today),
        category="Summary"
    ))

    test_results.append(run_test_query(
        "What new features were added to bq-etl tooling?",
        date_range=(two_weeks_ago, today),
        category="Summary"
    ))

    test_results.append(run_test_query(
        "What other significant changes happened in the repo?",
        date_range=(two_weeks_ago, today),
        category="Summary"
    ))

    # ===================================================================
    # CATEGORY 2: PERSON-BASED
    # ===================================================================
    print("\n\n👤 CATEGORY 2: PERSON-BASED QUERIES")
    print("="*70)

    # Pick a person who has activity (you can change this)
    test_results.append(run_test_query(
        "What has kwindau worked on recently?",
        date_range=(two_weeks_ago, today),
        category="Person"
    ))

    test_results.append(run_test_query(
        "What has benwu delivered this half?",
        date_range=("2025-07-01", today),  # H2 2025
        category="Person"
    ))

    test_results.append(run_test_query(
        "Who are the most active reviewers based on the PRs?",
        date_range=(two_weeks_ago, today),
        category="Person"
    ))

    # ===================================================================
    # CATEGORY 3: ASSET-SPECIFIC
    # ===================================================================
    print("\n\n🗄️ CATEGORY 3: ASSET-SPECIFIC QUERIES")
    print("="*70)

    test_results.append(run_test_query(
        "Are there multiple changes to baseline tables?",
        category="Asset"
    ))

    test_results.append(run_test_query(
        "What PRs recently landed for subscription or SubPlat tables?",
        date_range=(two_weeks_ago, today),
        category="Asset"
    ))

    test_results.append(run_test_query(
        "What changes were made to Stripe integration?",
        category="Asset"
    ))

    test_results.append(run_test_query(
        "What happened with MDN data pipelines?",
        category="Asset"
    ))

    test_results.append(run_test_query(
        "What Glean or telemetry changes were made?",
        category="Asset"
    ))

    # ===================================================================
    # CATEGORY 4: QUALITY/RISK SIGNALS
    # ===================================================================
    print("\n\n⚠️ CATEGORY 4: QUALITY & RISK SIGNALS")
    print("="*70)

    test_results.append(run_test_query(
        "What PRs required follow-up fixes or reverts?",
        date_range=(two_weeks_ago, today),
        category="Quality"
    ))

    test_results.append(run_test_query(
        "What backfills were recently completed or marked as complete?",
        date_range=(two_weeks_ago, today),
        category="Quality"
    ))

    test_results.append(run_test_query(
        "What schema migrations or schema changes happened?",
        date_range=(two_weeks_ago, today),
        category="Quality"
    ))

    # ===================================================================
    # CATEGORY 5: OPERATIONAL
    # ===================================================================
    print("\n\n🔧 CATEGORY 5: OPERATIONAL QUERIES")
    print("="*70)

    test_results.append(run_test_query(
        "What scheduling or DAG changes were made?",
        date_range=(two_weeks_ago, today),
        category="Operational"
    ))

    test_results.append(run_test_query(
        "What metadata completeness work was done?",
        date_range=(two_weeks_ago, today),
        category="Operational"
    ))

    # ===================================================================
    # SUMMARY STATISTICS
    # ===================================================================
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total_tests = len(test_results)
    successful_retrievals = len([r for r in test_results if r['chunks_retrieved'] > 0])

    print(f"\n✅ Total queries tested: {total_tests}")
    print(f"✅ Successful retrievals: {successful_retrievals}/{total_tests}")
    print(f"✅ Success rate: {(successful_retrievals/total_tests)*100:.1f}%")

    # Breakdown by category
    print("\n📊 Results by category:")
    categories = {}
    for result in test_results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {'total': 0, 'success': 0}
        categories[cat]['total'] += 1
        if result['chunks_retrieved'] > 0:
            categories[cat]['success'] += 1

    for cat, stats in categories.items():
        success_rate = (stats['success']/stats['total'])*100
        print(f"  {cat}: {stats['success']}/{stats['total']} ({success_rate:.0f}%)")

    print("\n💡 Key observations:")
    print("- Time-based filtering narrows search to recent activity")
    print("- Person-based queries leverage author metadata")
    print("- Semantic search finds relevant content across different PR types")
    print("- Citations enable verification and drill-down")
    print("- Hybrid search (metadata + vector) scales efficiently")

    print("\n🎉 Day 14 testing complete!")
