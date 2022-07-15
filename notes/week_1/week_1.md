# Week 1 Ranking & Relevance

## 20th June Class notes

Ranking & reranking approach (multi phase ranking)
- start simple
- add diversification/ business important 
- List Wide Ranking (LWR) Looks at all result pages (v. expensive)
- Learning to Rank (LTR)
  - index documents
  - train model
  - use inexpensive score & do a cut off
  - apply scoring algorithm you have trained and apply it to this top K
  - blend this and return it

Initialize LTR storage (tell the engine which features (ie the price) you want to use)

---

## Measuring Relevance

*“how well a retrieved document or set of documents meets the information need of the user”.*

We need to infer what the user wants through collecting judgments, either by manually assigning labels and classifying results or through implicit behavior.

- isolate queries that may be machine generated or relate to something else (such a specific field)

***REMINDER look at your query logs and granular data to understand where aggregations/ averages come from.***

### Important to log
- Query
- parameters
- session
- user
- results
- engagement
- system
- user input and autocomplete suggestion selected
- original query and spelling correction suggestions

### Key Metrics
- Precision (P)
  - position irrelevant
  - true positive / true positive + false positive
  - precision of the top 10 results, denoted P@10
  - average precision (weight given to the top ranked results)
  - works best with explicit human judgments
  - for behavior best to look at CTR or MRR
- Recall (R)
  - position irrelevant
  - true positive / false negatives + true positives
- Mean Reciprocal Rank (MRR)
  - for each query, take the reciprocal (the reciprocal of x is 1/x) of the position of the first relevant or clicked document and then take the mean across all queries.
  - works well for both explicit or implicit judgments
  - **Keep in mind that you also have to decide what to do with queries that have no relevant or clicked documents.** We recommend to treat their positions as infinity, the reciprocal of which is zero.
- Discounted Cumulative Gain (DCG)
  - rewards highly relevant results appearing at the top of the list
  - normalized variant NDCG
  - designed for explicit graded human judgments, it can also be used with implicit judgments if grades are based on behavior (e.g., a conversion is 10 times as good as a click).
- Search sessions (success)

### Vectors and Token weights to determine document similarity

* A cosine of 1 means that two vectors point in the same direction. Sorting results by cosine is a way of sorting by similarity to the query.
* Document size matters! So normalization is required.


Represented in an array of real number for each dimension or where vectors are sparse (many zero values) we can use a dictionary.

### Dot product - sum of products

$(||a|| = \sqrt{a ⋅ a})$

$cos(θ)=(a⋅b)/∣∣a∣∣∣∣b∣∣$

### Query dependant/ independent signals


| Query dependent                 | Query Independent (document boosts)  |
|---------------------------------|--------------------------------------|
|  # of matching tokens           | popularity                           |
|  tokens matching specific field | recency                              |
|  tf-idf, BM25 scores            | price                                |
|  synonyms                       | sale rank                            |

**Query-dependent signals are good for determining relevance. Query-independent signals are good for determining desirability.**


### Techniques to improve relevance in ranking
- hand tuning
- analyzers
- field and document boosting
- content understanding
- query expansion with synonyms
- auto phrasing (multi word input that can be treated as a phrase)
- query understanding
- learning to rank (LTR) - ML ranking
- pseudo relevance feedback (aka blind feedback)
- experimentation (AB tests)
- manual overrides
- user experience

### Pitfall in relevance tuning
- overfitting
- failure to iterate (take a agile approach)
- premature optimization
- relying on anecdotal data
- lack of tracking or infrastructure
- ignorance of tradeoffs
- pursuing diminishing returns
- tunnel vision (search queries take place in context of sessions)
- configuration issues (analyzers)

**Make sure you are collecting behavioral data for implicit judgments**

**Establish an ongoing query triage process.** On a regular cadence have your team each judge the results for the 50 most frequent queries, as well as for a random sample of less frequent queries.

**Measuring Document similarity:** We only need to worry about coordinates where both vectors have 1s, since all the other products will be 0. Which means that, for these vectors, the dot product just counts the number of tokens that the two documents have in common.

### tf-idf (tf*idf)

**tf (term frequency):** A token that is repeated in a document is more important to the document.  
**idf (inverse document frequency:** A token that occurs in fewer documents in the index is more important to the documents in which it occurs.  

## Relevance Engineering with OpenSearch

### Debugging relevance

#### How was my query parsed?

'explain=true'

```
GET /searchml_test/_validate/query?explain=true
{
  "query": {
      "term":{
        "title": "dog"
      }
  }
}
```
*When you are debugging queries, keep an eye out for fields that are missing, surprising boosts, and parsing exceptions that might explain why you have bad results.*

#### How was this document analyzed?

***Common error:*** indexing analyzer is not similar to the query processing analyzer

Analyze endpoint shows us the results of text analysis. Look for stemming issues, wrong offsets and positions, missing or unexpected tokens, etc. 

```
GET /_analyze
{
  "analyzer": "english",
  "explain": "true",
  "text": ["Wearing all red, the Fox jumped out to a lead in the race over the Dog.", "All lead must be removed from the brown and red paint."]
}
```

#### Why did this document match this query?

`Explain endpoint` lets us see why a particular document matched a particular query, or how its score was calculated.

- Start with the big picture: which terms and clauses are contributing the most to the score? Do those values look reasonable? Is there one clause that dominates the rest?
- Are all the fields present that you would expect to contribute to the score?
- Is the right similarity measure being used?
- How are boosts being distributed across the clauses and in the document?

#### Make your query logs searchable

Nested aggregations can also help, so first splitting by data and taking the top K per data then top K of those results.

## Multi-phase approach Ranking

- computational efficiency
- ideally rankers are aligned with each successive ranker simply being more precise.
- rankers can use completely different approaches
- can be used to address concerns like search result diversity or implement business rules

Improving relevance is mostly about better representing the signals from documents, queries, and users

**Index:** improve the way the content is represented, including by making better use of external resources. Includes content understanding.

**Query:** improve the way the query is represented, rewriting it or otherwise transforming it before using it for scoring. Includes query understanding.

**Retrieval:** since retrieval determines which results will be scored, make sure it’s doing a good job of balancing precision and recall before the ranker scores its results.

**Scoring:** this is what most people think of as the core of relevance. And indeed it’s important, especially after you’ve done what you can with the index, query, and retrieval.

**Think about how to solicit the best input from your users.**

### Implement multi-phase ranking

#### OpenSearch Rescoring

Use one approach, pick off the top K and rescore with another approach

## Altering results

- manually specified results (pinned queries)
- reranking through rules, scripts and custom scoring
- OpenSearch supports adding scripting (including in queries). Note that ths will slow down scoring!
- when adding rules make them easy to roll back, testable, record meta data like why they were created and log when and why they are fired (remove ones that are not being used)

## Learning to Rank (LTR)

- usually relies on supervised learning approach
- leverages training data with either explicit human judgement or implicit judgement (CTR)
- promotes relevant or desirable content over other results
- libraries include [XGBoost](https://xgboost.readthedocs.io/) and [Ranklib](https://sourceforge.net/p/lemur/wiki/RankLib/)

## LTR on OpenSearch 

### (step one) Initialize the LTR storage

### (step two) Populate the OpenSearch LTR feature store for the application with a feature set

A feature set contains one or more features, such as the sales rank or the text of a field.

### (step three) Collect judgments, either explicitly or implicitly

In order to obtain an implicit judgment for LTR, we need the following:

1. The user’s query.
2. The results shown to the user, with their corresponding positions.
3. Which of those results the user clicked on.

Ways we might achieve number 2 (results shown to the user, both that were clicked but also those which were not):

- Naively calculate the **probability of a document being clicked for a query** by dividing the number of clicks by the number of appearances in the results. This approach is simple but has two problems: it ignores position, and it is noisy for low counts.

- Model **whether a user saw a result based on what the user clicked on**, e.g., if the user clicked on a result, then the user saw all previous results.

- Gather **graded training data** based not only on whether the user clicked, but also the **dwell time** on the result, **downstream actions** like a purchase, etc.

⚠️ Beware

- clicks don't always tell us whether the user found something relevant, think about snippets that a user might read but not click
- be careful not to read too much into small data
- implicit click-based judgments suffer from positional bias
- presentation bias: you can only learn about results users see
- fine-grained judgments are necessarily better: bucketing the scores into a coarser-grained set of grades is a good way to avoid reading too much into meaningless differences.

### (step four) Join the features with the judgments by “logging the feature scores” to create a training data set
This involves executing queries to retrieve the defined features for each document, gathering the associated weights and writing out the results.

Our goal is to output records to a file formatted as follows:

`<grade> qid:<query_id> <feature_number>:<weight>... # <doc_id> <comments>`

### (step five) Train and test your model

- we can visualise the output of our model using matplotlib and graphviz

### (step six) Deploy your model to OpenSearch

POST to the `createmodel` endpoint

- though the model is uploaded under the `featureset` API, it isn’t owned by the `featureset`
- definition attribute is a single quoted and escaped string of `JSON` and **not** a Python dictionary
- the LTR plugin hasn’t kept pace with XGBoost deprecating `“reg:linear”` for `“reg:squarederror”`, so you can’t use that despite it being XGBoost’s default objective

### (step seven) Search with LTR in your application

```
query_obj["rescore"] = {
    "window_size": 10,
    "query": {
        "rescore_query": {
            "sltr": {
                "params": {
                    "keywords": queries[1]
                },
                "model": model_name,
                "store": ltr_store_name,
                # Since we are using a named store, as opposed to simply '_ltr', we need to pass it in
                "active_features": [title_query_feature_name, body_query_feature_name, price_func_feature_name]
            }
        },
        "rescore_query_weight": "2" # Magic number, but let's say LTR matches are 2x baseline matches
    }
}
response = client.search(body=query_obj, index=index_name)
```