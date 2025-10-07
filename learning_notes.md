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
- ⚠️ Inconsistent results due to using chat mode3l vs dedicated embedding model

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

### Sentence-Transformers Deep Dive

**Consistency vs Randomness:**
- Claude API: Different embeddings each run (chat model repurposing)
- sentence-transformers: Identical results every time (0.213 always = 0.213)
- ✅ **Key insight:** Dedicated embedding models are essential for production

**Understanding Cosine Similarity Scores:**
- **1.0** = identical meaning (same sentence)
- **0.8-1.0** = very similar meaning
- **0.5-0.8** = moderately similar  
- **0.2-0.5** = somewhat related (our "table saws vs carpentry" = 0.213)
- **0.0-0.2** = barely related (minivan vs data engineering = 0.000)
- **Negative** = opposite meanings (table saws vs Japan trip = -0.074)

**What Drives High Similarity:**
- **Specific shared concepts** get highest weight ("wellness day" appeared in both sentences → 0.434)
- **Domain clustering** works well (woodworking sentences cluster together)
- **Rare/specific terms** weighted more than common words
- **Semantic importance** over grammatical similarity

**Why This Enables Semantic Search:**
- Search "wellness activities" → finds both wellness day sentences
- Search "woodworking tools" → finds table saws, carpentry, and project sentences  
- No exact word matching required - meaning-based matching
- Perfect foundation for RAG systems that need to find relevant context

## Understanding Tokens - Key Learnings

**What Are Tokens?**
- **Smallest meaningful units** that AI models process
- **Not exactly words or characters** - they're linguistic building blocks  
- **Preserve meaning** - prevent splitting related concepts like "President" or "don't"

**Why Tokens Matter:**
- AI models have **token limits** (not word/character limits)
- Claude has ~200K token context window
- Embeddings work on token sequences
- Chunking by tokens ensures consistent processing

**Why Not Just Use Words?**
- "don't" = 2 tokens ("don" + "'t")
- "Mrs." = 2 tokens ("Mrs" + ".")
- Tokens handle punctuation, contractions, and subwords more precisely

**Token Math from Our Transcript Analysis:**
- **3,366 tokens** total in West Wing transcript
- **4.01 characters per token** average
- **~0.75 words per token** (3,366 tokens ≈ 2,500 words)
- **500 characters = 119 tokens**

**Why Characters Per Token Is Critical:**

1. **Planning & Estimation** - When you see content, you can estimate tokens:
   - Email (2,000 characters) ≈ 500 tokens
   - Meeting transcript (10,000 characters) ≈ 2,500 tokens

2. **Chunking Strategy** - Helps plan meaningful chunk sizes:
   - 500-token chunk ≈ 2,000 characters ≈ 2-3 paragraphs
   - 1000-token chunk ≈ 4,000 characters ≈ 1-2 pages

3. **Context Window Management**:
   - Claude's 200K tokens ≈ 800,000 characters of typical text
   - Your transcript (13,500 chars) uses only ~1.7% of Claude's context

## Token Limits & Context Windows (2025)

**What are tokens?** The smallest meaningful units that LLMs process - pieces of words, whole words, or punctuation. Context window = how many tokens a model can hold in memory at once.

| Model Family | Model | Context Window (Tokens) | Notes |
|--------------|--------|------------------------|--------|
| **100M+ Token Tier** | | | |
| Magic.dev | LTM-2-Mini | 100M | Experimental, can process entire codebases |
| **10M Token Tier** | | | |
| Meta | Llama 4 | 10M | Massive context for enterprise use |
| **1M Token Tier** | | | |
| Anthropic | Claude Sonnet 4 | 1M | Upgraded from 200K in 2025 |
| Google | Gemini 2.5 Flash | 1M | Fast processing with huge context |
| Google | Gemini 2.5 Pro | 1M | Includes "Deep Think" reasoning |
| OpenAI | GPT-4.1 | 1M | Latest flagship model |
| Meta | Llama 4 Maverick | 1M | Open source alternative |
| **400K Tier** | | | |
| OpenAI | GPT-5 | 400K | 128K output window |
| **200K Tier** | | | |
| Anthropic | Claude Opus 4 | 200K | High precision workflows |
| Anthropic | Claude Sonnet 3.7 | 200K | Balanced performance |
| Anthropic | Claude Haiku 3.5 | 200K | Cost-efficient |
| Alibaba | Qwen | 258K | Extendable to 1M |

**Key Insights:**
- **Massive growth in 2025:** Standard context windows jumped from 200K to 1M+ tokens
- **1M tokens ≈ 750,000 words ≈ 3,000 pages of text**
- **Why chunking still matters:** Even with 1M token windows, you might have 50M+ tokens of documents to search through
- **RAG is essential:** Can't fit everything in context, so chunk and retrieve relevant pieces

**Sources:** [GitHub LLM Context Limits](https://github.com/taylorwilsdon/llm-context-limits) | [2025 LLM Comparison](https://redblink.com/llm-comparison-chatgpt-gemini-grok-claude-deepseek/)

## Chunking Strategy Fundamentals

**What is a chunk?** A piece of your larger document - like breaking a book into chapters or a meeting transcript into segments.

**Why chunk at all?**
- Embeddings work better on focused content  
- Better retrieval precision (find specific topics, not entire documents)
- Context window management (even 1M tokens has limits vs 50M+ token datasets)

### Chunking Trade-offs Discovered:

**100-token chunks:**
- ❌ Breaks mid-conversation
- ❌ Lost context and incomplete thoughts
- ✅ Very focused topics (when not broken)

**500-token chunks:**  
- ✅ Complete conversations captured
- ❌ Multiple scenes/topics mixed together
- ⚖️ Reasonable compromise for most use cases

**1000-token chunks:**
- ✅ Very complete context preservation
- ❌ Way too many different scenes/topics per chunk
- ❌ Poor for specific topic retrieval

### Key Insight: The Chunking Dilemma
**There's no perfect chunk size** - it depends on your use case:

- **Small chunks (100-200 tokens):** Good for very specific queries, bad for context
- **Medium chunks (400-800 tokens):** Balanced approach, most common choice  
- **Large chunks (1000+ tokens):** Good for context, bad for precision

**Example from West Wing transcript analysis:** 500 tokens appeared to be the sweet spot - captures complete conversations without excessive topic mixing.

**Key principle:** Always test retrieval quality with embeddings to validate chunking strategy for your specific content type.

## Retrieval Pipeline: How Text Becomes Searchable

**The complete flow from text to search:**

1. **Text → Tokens:** Break text into linguistic building blocks
   - Tokens are based on common patterns, not strict word rules
   - Example: "President" = 1 token, "Presidente" = 2 tokens (less common in English)
   - Average: ~4 characters per token in English

2. **Tokens → Chunks:** Group tokens into meaningful segments
   - Use token count to create chunks (100, 500, 1000 tokens)
   - Chunk size determines precision vs context trade-off

3. **Chunks → Embeddings:** Convert to numerical vectors
   - Use embedding model (e.g., sentence-transformers all-MiniLM-L6-v2)
   - Output: Fixed-dimension vectors (384 dimensions for all-MiniLM-L6-v2)
   - **Critical:** Dimension count is model-specific, NOT input-dependent
     - 100-token chunk → 384 dimensions
     - 500-token chunk → 384 dimensions
     - 1000-token chunk → 384 dimensions
     - Even 5-word query → 384 dimensions
   - **Model comparison (you have choices!):**
     - sentence-transformers all-MiniLM-L6-v2: 384 dimensions
     - OpenAI text-embedding-ada-002: 1,536 dimensions
     - Google Universal Sentence Encoder: 512 dimensions
     - Different models = different dimension counts, can't mix them!

4. **Query → Embedding:** User's question gets same treatment
   - **Must use same model** as chunks (dimension count must match!)
   - Can't compare 384-dim embedding with 1,536-dim embedding (different spaces)

5. **Similarity Matching:** Find most relevant chunks
   - Use cosine similarity to compare query embedding vs all chunk embeddings
   - Score range: -1 (opposite) to 1 (identical)
   - In practice: most text similarities are 0 to 1 (negatives are rare)
   - Highest score = most relevant chunk

**Why dimensions must match:** Embeddings exist in dimensional space. A 384-dimension vector is a point in 384-dimensional space. A 1,536-dimension vector is a point in 1,536-dimensional space. You can't calculate distance between points in different dimensional spaces - the math doesn't work.

## Day 2 Completion: Chunking + Retrieval Testing

**Key insight: Chunk size is an art, not a science.**

### Experimental Results (West Wing Transcript - 3,366 tokens)

Query: "Who is Toby?"
- **100-token chunks:** 0.301 similarity (BEST)
- **500-token chunks:** 0.176 similarity
- **1000-token chunks:** 0.176 similarity

**Why 100-token chunks won:**
- Toby mentioned multiple times in small chunk = 60-80% of chunk content
- Same mentions in 500-token chunk = only 10-15% of content (diluted)
- Smaller chunks = higher topic density for specific queries

**The Trade-off:**
- **Small chunks (100 tokens):** Best for specific, factual queries ("Who is X?", "What is Y?")
- **Large chunks (1000 tokens):** Best for broad, contextual queries ("What happened?", "Summarize the discussion")
- **Medium chunks (500 tokens):** Balanced compromise

**Decision Framework:**
1. What questions will you ask? (Specific vs broad)
2. What's your content structure? (How many topics per document?)
3. What's your processing budget? (More chunks = more cost)

**"Ask George" Decision:** 1000-token chunks
- Typical meeting: 30 minutes = ~3,366 tokens = 3-4 chunks
- Meeting structure: 1-3 topics = ~1 chunk per topic
- Query style: Mostly broad contextual ("What was discussed about X?")
- Processing efficiency: 3-4 chunks per meeting vs 34 chunks at 100 tokens

**Remember:** There's no universal "best" chunk size. Test with your actual content and query patterns.

## Day 3: ChromaDB - Vector Database Fundamentals

**Code:** [understanding_chromadb.py](./understanding_chromadb.py)

### Why Use a Vector Database?

**Problem with Day 2 approach (numpy arrays):**
- Re-compute embeddings every time script runs
- Load everything into memory
- Manual similarity search
- No persistence between runs

**ChromaDB solves this:**
- ✅ **Persistence:** Generate embeddings once, store forever
- ✅ **Auto-embedding:** Handles query embedding automatically
- ✅ **Optimized search:** Fast similarity search even with thousands of chunks
- ✅ **Scalable:** Add hundreds of meetings without memory issues

### Key Concepts Learned

**Collections:**
- Like containers for different data types
- Example: `meeting_transcripts`, `emails`, `google_docs`
- Each collection uses the same embedding model for consistency

**ChromaDB's Default Embedding Model:**
- Uses **all-MiniLM-L6-v2** (same as Day 1 sentence-transformers!)
- 384 dimensions
- Downloads automatically on first use

**Distance vs Similarity:**
- **Similarity (Day 2):** 0.0 to 1.0, higher = better (cosine similarity)
- **Distance (ChromaDB):** 0.0+, lower = better (inverse concept)
- Same idea, different metric

### Experimental Results

**4 chunks (1000 tokens each) from West Wing transcript**

Query comparison:
- "Who is Toby?" → Distance: 1.649 (weak match - just name mention)
- "What was discussed about the president?" → Distance: 1.348 (BEST - keyword overlap!)
- "Tell me about military or battleships" → Distance: 1.706

**Key Insight:** Semantic search still benefits from keyword overlap. If "president" appears frequently in a chunk, that chunk scores well for queries containing "president". It's not pure magic - it's enhanced keyword matching with semantic understanding.

### ChromaDB Basic Operations

```python
import chromadb

# Create client and collection
client = chromadb.Client()
collection = client.create_collection(name="meeting_transcripts")

# Add documents (auto-embeds them)
collection.add(
    documents=["chunk text 1", "chunk text 2"],
    ids=["chunk_0", "chunk_1"]
)

# Query (auto-embeds query)
results = collection.query(
    query_texts=["What was discussed?"],
    n_results=3
)
```

**What's happening behind the scenes:**
1. Documents → Embedded with all-MiniLM-L6-v2 → Stored with 384-dim vectors
2. Query → Embedded with same model → Compared to all stored vectors
3. Returns closest matches by distance

## Day 4: Metadata Filtering for Contextual Queries

**Code:** [understanding_chromadb.py](./understanding_chromadb.py) (updated)

### What is Metadata?

**Metadata** = Information *about* the document/chunk, known before reading content
- Examples: date, participants, meeting type, document source

**Content** = What's *inside* the text (discovered by reading/searching)
- Examples: budget amounts, decisions made, action items

### Why Metadata Matters

**Without metadata:** "What was discussed about the budget?" → searches ALL meetings

**With metadata:** "What was discussed about the budget in October team meetings with Alice?" → filters first, then searches

**This is hybrid search:** Filter by structured data + semantic search by meaning

### Metadata Strategy for "Ask George"

**Chosen metadata fields:**
- `date`: Filter by time period
- `meeting_title`: Identify recurring meetings
- `participants`: Filter by people (MOST VALUABLE - participants proxy for topic/project)
- `meeting_type`: Distinguish 1-on-1 vs team meetings

**Why participants is most valuable:** Different people = different work areas. Filtering by participants quickly narrows to relevant context.

### ChromaDB Metadata Implementation

```python
# Add metadata to chunks
collection.add(
    documents=["chunk text 1", "chunk text 2"],
    ids=["chunk_0", "chunk_1"],
    metadatas=[
        {"date": "2024-10-01", "meeting_type": "team", "participants": "Alice, Bob"},
        {"date": "2024-10-01", "meeting_type": "team", "participants": "Alice, Bob"}
    ]
)

# Query with metadata filter
results = collection.query(
    query_texts=["What was discussed about budget?"],
    where={"meeting_type": "team"},  # Only search team meetings
    n_results=3
)
```

**Filter logic:**
1. Filter to chunks matching metadata criteria
2. Run semantic search within filtered set
3. Return best matches

**Key insight:** All chunks from the same meeting share the same metadata (metadata describes the meeting, not individual chunks).

## Day 5: End-to-End RAG Pipeline with Claude API

**Code:** [understanding_rag.py](./understanding_rag.py)

### What is RAG (Retrieval Augmented Generation)?

RAG = Connecting retrieval (vector database) with generation (LLM) to answer questions based on your specific knowledge base.

**The RAG Flow:**
1. User asks question
2. Retrieve relevant chunks from vector database
3. Build prompt with context + question
4. LLM generates answer based ONLY on provided context

### RAG vs Non-RAG Comparison

**Question:** "Who is Toby?"

**Without RAG (Claude alone):**
- Asked for clarification
- Suggested multiple possibilities: Toby from The Office, Pretty Little Liars, etc.
- No access to your specific data

**With RAG (Claude + retrieved context):**
- "Toby Ziegler is introduced to Carl Everett, works with Sam and Josh..."
- Answer grounded in YOUR transcript
- No hallucination - stuck to provided context

### Key RAG Concepts Learned

**1. Context Window Management:**
- Retrieved 2 chunks = 8,011 characters of context
- Sent as part of prompt to Claude
- Context + question + instructions all fit in Claude's context window

**2. Prompt Engineering for RAG:**
```
You are a helpful assistant that answers questions based on provided context.

Context from meeting transcripts:
[Retrieved chunks here]

Question: [User's question]

Please answer based ONLY on the information in the context above.
If the context doesn't contain enough information, say so.
```

**3. Conversation History (Multi-turn):**
- Each API call is stateless
- To maintain conversation, include previous messages:
  ```python
  messages=[
      {"role": "user", "content": "First question with context"},
      {"role": "assistant", "content": "First answer"},
      {"role": "user", "content": "Follow-up question"}
  ]
  ```

### Critical Insight: Grounding Prevents Hallucination

**The power of RAG:** Forces LLM to answer from YOUR data only
- Without context: Claude guesses or asks for clarification
- With context: Claude answers from provided information or admits "I don't know"
- No making things up based on training data

**This is why RAG is essential for:**
- Personal knowledge bases (meetings, docs, emails)
- Company-specific information
- Private data that wasn't in LLM training
- Any domain requiring factual accuracy from specific sources

## Day 6: Improving RAG with Citations and Testing

**Code:** [understanding_rag.py](./understanding_rag.py) (updated)

### Adding Citations to RAG

**Why citations matter:**
1. **Verify accuracy** - Check if Claude's interpretation matches source
2. **Enable follow-ups** - Know which meeting to dig deeper into
3. **Build trust** - Users can see where answers come from

**Implementation approach:**
- Include metadata in context alongside chunk text
- Add prompt instruction: "Cite which source(s) you used"
- Format: `Source {i+1} (Meeting Title, Date): [chunk text]`

**Example citation prompt:**
```
Source 1:
- Meeting: West Wing Strategy Session
- Date: 2024-10-01
- Participants: Toby, Sam, Josh
- Content: [chunk text]

Question: Who is Toby?

Please answer based ONLY on the information in the sources above.
At the end of your answer, cite which source(s) you used.
```

**Result:** Claude now answers + adds "**Source: West Wing Strategy Session, 2024-10-01**"

### Testing Different Question Types

**Test questions:**
1. "Who was flirting with who?" (relationship detection)
2. "What was the impasse with the labour union?" (specific facts)
3. "Was the hurricane response effective?" (requires judgment)
4. "Who won the monday night football game?" (not in transcript)

### What Works Well vs What Doesn't

**✅ Works Well: Direct Factual Questions**
- "Who was flirting?" → Found Danny & C.J., cited dialogue as evidence
- Questions about who, what, when, where
- Information directly stated in text
- Can be answered without external knowledge

**⚠️ Limited: Judgment/Opinion Questions**
- "Was the hurricane response effective?" → Honestly said "cannot determine"
- Listed what WAS mentioned (FEMA call, communication issues)
- But couldn't evaluate effectiveness without external criteria
- Requires evaluation framework not in the context

**✅ Works Well: Honest "I Don't Know"**
- "Monday Night Football winner?" → Clear "cannot answer based on sources"
- No hallucination when info isn't available
- RAG successfully prevents making things up

### Key Insights

**RAG's Strengths:**
- Retrieval of factual information from your data
- Citing sources for verification
- Honest admission when information is missing
- No hallucination on out-of-scope questions

**RAG's Limitations:**
- Can't apply external knowledge or judgment
- Can't evaluate "effectiveness" without criteria
- Limited to what's explicitly in the retrieved context
- Requires factual content, not just tangentially related text

**The fundamental trade-off:** RAG grounds answers in your data (good for accuracy) but sacrifices the LLM's broader reasoning capabilities (limits judgment questions).
