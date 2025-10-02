Embeddings

[What are embeddings and how do they work?](https://medium.com/@eugenesh4work/what-are-embeddings-and-how-do-it-work-b35af573b59e)

## Day 1-2: Understanding Embeddings - Hands-on Experience

**Code:** [understanding_embeddings.py](./understanding_embeddings.py)

### Key Learning Takeaways

**1. What are embeddings?**
- Convert text into numerical vectors (coordinates in high-dimensional space)
- Each dimension captures different semantic aspects  
- Similar meanings → similar vectors → higher cosine similarity

**2. Why embeddings enable semantic search?**
- Traditional search: exact word matching
- Embedding search: meaning matching
- 'data pipeline' and 'ETL job' can be found as similar even with no shared words

**3. What we learned from this experiment:**
- ✅ Successfully generated embeddings using Claude API
- ✅ Calculated cosine similarities between sentence pairs
- ✅ Understood the mathematical foundation
- ⚠️ Inconsistent results due to using chat model vs dedicated embedding model

**4. Key insight from results:**
- Some semantic clustering worked (woodworking sentences clustered together)
- But many results were inconsistent or unexpected
- This demonstrates why purpose-built embedding models exist!

**5. For real applications, use:**
- OpenAI text-embedding-ada-002
- sentence-transformers library  
- Google Universal Sentence Encoder
- These give consistent, reliable semantic representations

**6. Foundation concepts mastered:**
- 📚 Embeddings = text → numerical vectors
- 📚 Cosine similarity = measure of vector closeness
- 📚 Semantic search = finding meaning, not just words
- 📚 This enables: search, recommendations, clustering, classification
