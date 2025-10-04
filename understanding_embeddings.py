from dotenv import load_dotenv
import os
import anthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

sentences = [
    "table saws are great but circular saws offer more flexibility",
    "I want to buy a minivan",
    "I haven't spent a lot of time doing carpentry this summer",
    "Growing grass in fall is great because it's cooler",
    "data engineering is fun until folks pull you into projects last minute",
    "I am sad I didn't get to go to Japan",
    "we need to do integration work for tapclicks reporting tool",
    "I should find a woodworking project for wellness day",
    "Mozilla's wellness day is coming up"
  ]


def get_embedding(client, text):
    """Get embedding for a single text using Claude API"""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": f"""Convert this text into a numerical vector that represents its semantic meaning.

                Text: "{text}"

                Return exactly 50 numbers between -1 and 1, separated by commas. Each number should capture 
different aspects of the meaning (topic, sentiment, concepts, etc.).

                Return only the numbers, no other text."""
            }]
        )

        # Parse response into array
        numbers_text = response.content[0].text.strip()
        print(f"Claude returned: '{numbers_text}'")
        numbers = [float(x.strip()) for x in numbers_text.split(',') if x.strip()]

        #Ensure we always have exactly 50 dimensions
        if len(numbers) < 50:
            # Pad with zeros if too short
            numbers.extend([0.0] * (50 - len(numbers)))
        elif len(numbers) > 50:
            # Truncate if too long
            numbers = numbers[:50]

        return np.array(numbers)

    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        return None


def get_sentence_transformer_embedding(model, text):
    """Get embedding using sentence-transformers (proper embedding model)"""
    return model.encode(text)


## Claude Embeddings
# Generate embeddings for a single sentence
test_embedding = get_embedding(client, sentences[0])
print(f"Embedding shape: {test_embedding.shape}")
print(f"First 5 values: {test_embedding[:5]}")

# Generate embeddings for all sentences
print("Generating embeddings for all sentences...")
embeddings = []
for i, sentence in enumerate(sentences):
    print(f"Processing sentence {i + 1}/{len(sentences)} : {sentence}")
    embedding = get_embedding(client, sentence)
    if embedding is not None:
        embeddings.append(embedding)
    else:
        print(f"Failed to get embedding for sentence {i + 1}")

embeddings = np.array(embeddings)
print(f"Generated {len(embeddings)} embeddings")

# Calculate similarity matrix
similarity_matrix = cosine_similarity(embeddings)
print("Similarity matrix shape:", similarity_matrix.shape)

# Show similarities between sentences
print("\nSentence similarities:")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        similarity = similarity_matrix[i][j]
        print(f"[{similarity:.3f}] '{sentences[i]}...' vs '{sentences[j]}...'")

## Sentence Transformer Embeddings
# Add this test after your existing code
print("\n" + "="*60)
print("SENTENCE-TRANSFORMERS COMPARISON")
print("="*60)

# Load a good general-purpose model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading sentence-transformers model...")
print("Generating embeddings with sentence-transformers...")

# Generate embeddings with sentence-transformers
st_embeddings = []
for sentence in sentences:
    embedding = get_sentence_transformer_embedding(st_model, sentence)
    st_embeddings.append(embedding)
    print(".", end="", flush=True)

st_embeddings = np.array(st_embeddings)
print(f"\nSentence-transformers embedding shape: {st_embeddings.shape}")

# Calculate similarities
st_similarity_matrix = cosine_similarity(st_embeddings)

# Compare a few key similarities
print("\nKey comparisons:")
print("Table saws vs Carpentry:")
print(f"  Sentence-transformers: {st_similarity_matrix[0][2]:.3f}")
print(f"  Claude: {similarity_matrix[0][2]:.3f}")

# Show all sentence similarities with sentence-transformers
print("\nAll sentence similarities (Sentence-Transformers):")
print("-" * 80)
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        st_similarity = st_similarity_matrix[i][j]
        print(f"[{st_similarity:.3f}] '{sentences[i][:45]}...' vs '{sentences[j][:45]}...'")

# Find the top 5 most similar pairs
print("\nTOP 5 MOST SIMILAR PAIRS:")
print("-" * 40)
similarities = []
for i in range(len(sentences)):
    for j in range(i+1, len(sentences)):
        similarities.append((i, j, st_similarity_matrix[i][j]))

similarities.sort(key=lambda x: x[2], reverse=True)

for idx, (i, j, score) in enumerate(similarities[:5]):
    print(f"{idx+1}. [{score:.3f}] '{sentences[i]}'")
    print(f"              vs '{sentences[j]}'")
    print()
