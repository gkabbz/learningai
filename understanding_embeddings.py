from dotenv import load_dotenv
import os
import anthropic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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