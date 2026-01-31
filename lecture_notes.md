---
id: Retrieval-Augmented
date: 2025-12-21
type: concept
cssclasses:
  - cornell-note
---
🔎**TL;DR:** 
- A *Retrieve first, then generate* LLM
- 🏷️**Tags:** #12-2025 #concept #2025-W50
---

> [!ccue] LLM - pattern-completer
- Given text, it predicts the *likely* next word.
- ~={orange}Limits:=~
	- It can't reliably recall private or latest info
	- Hallucination
	- Fix context window

## Retrieval-Augmented Generation

> [!ccue] RAG
- **Search first, then write**, using the retrieved documents as evidence.
- ~={green}Advantages=~:
	- Accuracy & grounding
	- Freshness
	- Traceability 
	- Cost/control
### Core components

![[Pasted image 20260103124955.png]]


> [!ccue] Embedding Models :LiBrain:
- Goal: To convert the info from text to a list of numbers that represents properties of the given info.
	- Infos with *similar* properties (e.g. same meaning) are close to each other in the vector space.
- Two types of info that need to be embedded:
	- **Document**: offline.
	- User's **query**: online.
```python
from langchain_openai import OpenAIEmbeddings
```

>[!ccue] Splitter :LiTableRowsSplit:

- In spite of increasing capacity of encoding models (e.g. 512 -> 8K tokens), there will be always a limited cap for it. Hence, encoding models might be unable to process a whole document, let's say a bible, so chunking - decomposing the document - into multiple sub-documents is necessary.
- Chunks of the same document are not necessarily independent of each other. Most of the time, we might want to *overlap* them.
- Criteria:
	- Delimiters (`.`, `;`, `!`, etc.)
	- Paragraph (`/n`)
	- Section
	- ...
- Example code:
	```python
	from langchain.text_splitter import RecusiveCharacterTextSplitter
	
	# docs = loader.load()
	
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
	splits = text_splitter.split_documents(docs)
	```

> [!ccue] Vector store :LiDatabase:
- A database that stores **info in vector form** and **meta-data** (e.g., access right, author, date), optimizing for retrieving relevant info via comparing similar vectors.
- This helps filter info quickly:
	- "give me ~={yellow}top-k=~ chunks closet to this query vector"
	- "only docs the user can access"
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(documents=splits,
									embeddings=OpenAIEmbeddings()) # <-- Embedding models which convert text to its numerical representation
```

> [!ccue] Retriever :LiFileSearch2:
- The *librarian* who can search for the *"relevant"* information based on user's requests.
- Two common retrieval types:
	- **Lexical** retrieval: Search based on keyword matching, e.g. BM25 / inverted index.
	- **Semantic** retrieval: Search based on the underlying meaning, but embedding models are required. 
- A retriever can opt for either one of two types or combine both. In either-or case, it's fast and simple
```python
retriever = vectorstore.as_retriever()
similar_docs = vector_store.similarity_search("your query here")
```

> [!ccue] Generator 🦜
- A LLM that generates, constrained by the retrieved documents.
	- $P(w_{n} w_{n-1} \dots w_{0} | \textit{ retrieved documents})$
- *"Based on the retrieved document, generate the answer; if the document does not contain it, say so."*

### Query Translation
Semantic retrieval is hard. Mainly from human-side with bad queries (e.g. ambiguous, badly worded, extremely concise, etc.). As we already know, for every AI model:

 > [!QUOTE] A notorious ML quote
 > Garbage in, garbage out.

> [!ccue] Query Translation
-  Definition: To **improve user's query**, thereby improving the efficiency of the retrieval process.
- Approaches:
	- **Rewrite**: Multi-query, RAG rank Fusion
	- **Decomposition**: Least-to-Most
	- **Step-back**
	- **HyDE** 
#### Query Rewrite
To rephrase the original query in multiple ways, in the hope that, in general, many queries will work better than a single query. Additionally, it can also help capture multiple nuances of the user's query.

![[Pasted image 20260104161354.png]]
> [!ccue] Multi-Query

A simple realization to rewrite query is to have a **in-the-middle chatbot** to handle the rewritting process, generating multiple perspectives on the user question. For example:
```python
from langchain.prompts import ChatPromptTemplate

# Multi Query: Different Perspectives
template = """You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. Original question: {question}"""
prompt_perspectives = ChatPromptTemplate.from_template(template)

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

> [!ccue] RAG-Fusion

With the retrieved relevant documents (splits), taking the union of the list might not go well as we thought. The generated *different-perspective* queries might retrieve *weakly* relevant documents. By flushing all documents into the prompt for the LLM to generate the final answer is, sometimes, too wasteful and risky. Noticeably, all retrieved documents must be ~={red}re-ranked :LiSortDesc: =~in order to utilize the most relevant documents only. Such approach is called **~={yellow}RAG-Fusion=~**. Example:
```python
from langchain.load import dumps, loads

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})
len(docs)
```
#### Decomposition
The user's query can also be decomposed into multiple sub-queries, or sub-problems, which can be solved sequentially. Prompt example:
```python
# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related to: {question} \n
Output (3 queries):"""
prompt_decomposition = ChatPromptTemplate.from_template(template)
```

~={yellow}**Recursive Answering**=~([[2212.10509] Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions](https://arxiv.org/abs/2212.10509)), reaching to the final answer in an interleaving scheme with 
![[Pasted image 20260104163736.png]]

```python
# Prompt
template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question: 

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template) ```
- Least-to-Most: https://arxiv.org/pdf/2205.10625.pdf
#### Step-back
In addition to the original query, retrieving relevant documents using a more abstract generated query. This one is beneficial for scenarios like documents with abstract, general information then followed by detailed technical information. Example:
```python
# Response prompt 
response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant. # {normal_context} # {step_back_context} # Original Question: {question} # Answer:""" 

response_prompt = ChatPromptTemplate.from_template(response_prompt_template)
```
- [arxiv.org/pdf/2310.06117](https://arxiv.org/pdf/2310.06117)
#### HyDE
User's query harbors inherent differences from stored documents, leading to a unnecessarily far distance between the query embeddings and the actual relevant document embeddings. These differences might come from the way of wording, the number of words, etc.

![[Pasted image 20260104165244.png]]
Solution: Leveraging chatbot to generate a Hypothetical Document which brings similarities with the stored documents. The hypothetical document is then embedded and used for similarity searching in the vector store. The more we know, the more detailed the prompt used for generating HyDoc, thus the HyDoc itself, can be. Example: 
```python
# HyDE document generation
template = """Please write a scientific paper passage to answer the question
Question: {question}
Passage:"""

prompt_hyde = ChatPromptTemplate.from_template(template)
```

## Routing
### Logical Routing
Let's the LLMs choose the correct ~={yellow}database=~ for retrieval
![[Pasted image 20260131142224.png]]

Code diagram:
![[Pasted image 20260131142325.png]]
### Semantic Routing
Let's the LLMs choose the appropriate ~={yellow}**prompt**=~ template to act:
![[Pasted image 20260131142244.png]]

## Query Construction

> [!ccue] Type of data

Our datastore appears in various types of data (*structured*, *unstructured*, and *semi-structured*). Although normal retrieval on processed query can perform adequately fine on average between natural language (*unstructured*) and other data types, each type of data has specific challenges and considerations:

| Structured                                                                                                                                                                                       | Unstructured                                                                                                                                                                      | Semi-Structured                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Predominantly housed within SQL or Graph databases, structured data is characterized by predefined schemas and organized in tables or relations, making it amenable to precise query operations. | Typically stored in vector databases, unstructured data consists of information without a predefined model, frequently accompanied by structured metadata that enables filtering. | Semi-structured data blends structured elements (e.g., tables in a document or relational database) with unstructured elements (e.g., text or an embedding column in a relational database). |

> [!ccue] Query Construction

To address these challenges, LLMs have great capacity for **query construction**, converting natural language into a specific formal syntax for each type of data. 

> Many user queries are best answered not just by finding documents or data similar in the embedding space, but by ~={yellow}taking advantage of the structure inherent=~ in the data and expressed in the user query.

For example, a query `what are the documents about Matrix Multiplication for fresher training, written by AI department`. We want to look up for Matmul-related documents semantically, but also a component about the category `training` and the author `AI department` that we might want to utilize to filter out unnecessary documents. Such kind of properties (category, author) live commonly in a structured DB, e.g. a SQL or a Graph database, as meta-data.

> [!ccue] Text-to-Metadata filters

![[Pasted image 20260131145546.png]]

Metadata Object:
```python
import datetime
from typing import Literal, Optional, Tuple
from langchain_core.pydantic_v1 import BaseModel, Field

class TutorialSearch(BaseModel):
    """Search over a database of tutorial videos about a software library."""

    content_search: str = Field(
        ...,
        description="Similarity search query applied to video transcripts.",
    )
    title_search: str = Field(
        ...,
        description=(
            "Alternate version of the content search query to apply to video titles. "
            "Should be succinct and only include key words that could be in a video "
            "title."
        ),
    )
    min_view_count: Optional[int] = Field(
        None,
        description="Minimum view count filter, inclusive. Only use if explicitly specified.",
    )
    max_view_count: Optional[int] = Field(
        None,
        description="Maximum view count filter, exclusive. Only use if explicitly specified.",
    )
    earliest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Earliest publish date filter, inclusive. Only use if explicitly specified.",
    )
    latest_publish_date: Optional[datetime.date] = Field(
        None,
        description="Latest publish date filter, exclusive. Only use if explicitly specified.",
    )
    min_length_sec: Optional[int] = Field(
        None,
        description="Minimum video length in seconds, inclusive. Only use if explicitly specified.",
    )
    max_length_sec: Optional[int] = Field(
        None,
        description="Maximum video length in seconds, exclusive. Only use if explicitly specified.",
    )
```

Database query converter:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """You are an expert at converting user questions into database queries. \
You have access to a database of tutorial videos about a software library for building LLM-powered applications. \
Given a question, return a database query optimized to retrieve the most relevant results.

If there are acronyms or words you are not familiar with, do not try to rephrase them."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(TutorialSearch)
query_analyzer = prompt | structured_llm
```

> [!ccue] Text-to-SQL

![[Pasted image 20260131150140.png]]
> [!ccue] Text-to-SQL + semantic

#TODO

> [!ccue] Text-to-Cypher

While vector stores readily handle the similarity search between queries and documents (or splits of docs), it doesn't understand the relationship between vectors, or, in other words, the relationship between documents. While the SQL database can expose these relationships to some extent with keys or table linkages, it becomes unnecessarily costly when there's a change in schema. On the other hand, graph databases don't suffer from those challenges when expressing documents' relationships.

Graph databases often use a specific query language called Cypher, focusing on providing a visual way of matching patterns and relationships. For example: `(:Person {name:"Tomaz"})-[:LIVES_IN]->(:Country {name:"Slovenia"})`. We can translate natural language to Cypher query to benefit from such inherent structure (name, LIVES_IN, etc.).
```Python
from langchain.chains import GraphCypherQAChain

graph.refresh_schema()

cypher_chain = GraphCypherQAChain.from_llm(
    cypher_llm = ChatOpenAI(temperature=0, model_name='gpt-4'),
    qa_llm = ChatOpenAI(temperature=0), graph=graph, verbose=True,
)

cypher_chain.run( "How many open tickets there are?" )
```

> [!ccue] Self-query retriever

#TODO
