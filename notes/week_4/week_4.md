# Week 4

## Class notes

- Tuesday Fire side with Dave Lewis (litigation (eDiscovery), corporate investigations, regulatory and information governance.)
- Wednesday project kick off
- Friday community share out
- submit 3/4 projects to get a certificate.

### Limitations of inverted index

- struggles with understanding sentiment and the similarity of meaning between sentences based on it's words (tokens)

### Vector search

- computer relevancy from query to document

In long queries, we might not know which words can be removed, or how to expand a query. However inverted index can sometimes do very well, ie looking for specific information. Vector search is good at general topic but not for fine grain control and additionally is slower than inverted index.

#### Representing documents as vectors

- get a single string that is normalized out into language model to create vector
- check out hugging face course on tokenization
- unlikely you train a model from scratch, more likely to fine tune the final layers of a pre built one

#### Representing Queries as vectors

- can use same model if the queries look similar (same normalization, same vocabulary, same style) as documents
- use two tower model
- aggregate on the vectors, to move query vector to same vector space as document. Good for head and torso queries, hard for tail queries

#### Retrieval: nearest neighbor search

- approximate nearest neighbor search, hierarchal small worlds (HNSW)
- 

**Search is not only indexing, ranking, query understanding etc**

### Project 4

- KNN plug in for OpenSearch
- Hugging face sentence transformers
- 

Query independent factors often require some normalization (ie price), if some overlap think about combining them to create features.

Query dependent factors, look at individual field and try to find what 

### References

- [AI for Query Understanding](https://dtunkelang.medium.com/ai-for-query-understanding-d8c073095fff)
- [Hugging Face course](https://huggingface.co/course/chapter0/1?fw=pt)
- [Semantic Product search](https://arxiv.org/pdf/1907.00937.pdf)
- [Extreme multi-label learning for semantic matching in product search](https://arxiv.org/abs/2106.12657)
- [Evaluating search](https://dtunkelang.medium.com/evaluating-good-search-part-i-measure-it-5507b2dbf4f6)
- [Introduction to Information Retrieval (2008)](https://nlp.stanford.edu/IR-book/information-retrieval-book.html)
- [Modern Information Retrieval](https://www.amazon.com/Modern-Information-Retrieval-Concepts-Technology-dp-0321416910/dp/0321416910/ref=dp_ob_title_bk)


## Vector Search

*"represents both the content and queries as vectors, typically through the use of word embeddings, and then retrieves and ranks results based on a distance or similarity function that describes the relationship between those vectors."*

- directly compute the relevance of a document to a query using representation of both that holistically represent their meaning
- traditional inverted index is challenged by words with multiple meanings (polysemy) which can result in a loss of precision, vector representation can, at least in theory, avoid this.
- synonyms and word variants can be addressed through techniques like stemming and dictionary-based query expansion, those don't scale well for tail queries. Vector representations are not subject to the "Vocabulary problem", multiple words that express the same meaning.
- inverted index applications can struggle with long queries, resorting to query relation to ensure recall but often at the expense of precision, in contrast vector search represents the query holistically so this precision-recall tradeoff doesn't happen

**However**

- vectors from word embeddings tend to be a lot less explainable than tokens, making them a bit of a black box
- a single embedding may not capture everything about a document or query and tend to be task dependant
- embedding vectors have large dimensions!
- vector search relies on an index designed for nearest-neighbor search, this is slower and less efficient
- vector search returns relevancy but not ranking, its hard to combine query dependent factors
- If we want to combine the output of a similarity search with facets, filters, or token-based retrieval, we're likely to incur a significant cost because of the different sort orders.

## Populating a Vector Search Index
