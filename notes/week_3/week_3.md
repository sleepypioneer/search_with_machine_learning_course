# Week 3 Query Understanding

*"transforming a search query to better represent the underlying search intent"*

- the kind of results the searcher is looking for
- the type of information need that the search is expressing (known-item-search versus exploratory)

## Query rewriting

Automatically transform search queries in order to better represent the searcher’s intent.

Increase recall, retrieve a larger set of relevant results, mostly we use query expansion and query relaxation
- **Query expansion**
	- broaden the query by adding ORs to the search query ie sneakers bag would become (sneakers OR shoe) AND bag
	- popular to do this with synonyms either from manual entry, thesauri, queries, content

	**Remember: The meaning of words always depends on context, so there is no such thing as a completely context-independent synonym pair.**
	
	- Matches using expanded tokens can be more relevant than only those using the original query and can therefore also increase precision.
- **Query relaxation**
	- instead of adding tokens to the query we remove them
	- increase recall by removing or making optional tokens that may not be necessary for relevance
	- can help show results that don't contain all of the original query tokens but have the same intent ie `microsoft windows laptop` and `windows laptop`
	- this might result in showing some results over showing none at all
	- [minimum_should_match](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-minimum-should-match.html) can help to ensure we don't remove too many tokens and loose relevancy
	- a more sophisticated approach would be to remove more common tokens (ie those which appear in most documents) but this is not fallible. For instance shoes will be a more common word but in the query `red shoes` we wouldn't want to remove shoes
	- another strategy is to parse the query grammatically and decide which tokens to make optional based on their parts of the speech, off shelf parsers can struggle with brands and for this we need named entity recognition (NER)
	- [part of speech tags](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html), parses are likely trained on complete sentences and will return unpredictable results for short search queries

	Mostly these techniques work well for queries which have few results but can be harmful to precision for queries which already have many results. However there are other query rewriting techniques that focus on improving precision.

## Query segmentation
- taking a segment of the query tokens and treating them as a phrase (so that they have to appear sequentially)
- [Pointwise mutual information (PMI)](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Pointwise_mutual_information): find token pairs that occur in sequence more often than independently, search logs could be used to do this
- another more simple approach would be to maintain a dictionary of known phrases

## Entity Recognition
- determines an entity type for each query segment, whether the segment contains a single token or multiple consecutive tokens
- Transform `“samsung phone”` to `Brand contains “samsung” AND Product Type contains “phone”`
- You can implement entity recognition with a dictionary-based approach
- suggested to build a query classifier first and then category specific entity recognition models to reduce complexity and to be able to build models incrementally

## Spelling correction
- 5-10% queries are misspelt
- [Aspell](https://www.google.com/url?q=https://en.wikipedia.org/wiki/GNU_Aspell), [Hunspell](https://www.google.com/url?q=https://en.wikipedia.org/wiki/Hunspell)
- [How to Write a Spelling Corrector](https://www.google.com/url?q=http://norvig.com/spell-correct.html)
- OFFLINE PROCESSES:
	- **Indexing tokens** - building out the index used at query time
		- maps substrings of tokens (character n-grams) to tokens
		- tradeoffs among storage, efficiency, and quality in indexing and candidate generation
		- it is possible to index tokens based on how they sound
	- **Build language model** 
		- model to estimate the [priori probability](https://corporatefinanceinstitute.com/resources/knowledge/other/a-priori-probability/)(probability derived from logically examining an event, it is the number of desireable outcomes divided by the total number of outcomes) of an intended query
		- we rely on frequency of n-gram frequencies for small values of n (ie bigrams, trigrams) for larger values of n we use backoff or interpolation this helps solve both the problem of exponential growth of token sequences to query length and sparsity (longer queries are more difficult to calculate probability in historical data)
	- **Build error model** - model to estimate the probability of a particular misspelling, given an intended query
		- common edits (spelling error corrections) include insertion, deletion, substitution, transposition (swapping consecutive letters)
- ONLINE PROCESS:
	- **Candidate generation** - identifying spelling correction candidates for the query from spelling correction index
	- **Scoring** - computing the score or probability for each candidate applying Bayes' theorem
		- $Prob(candidate | query) ∝ Prob (query | candidate) * Prob (candidate)$
	- **presenting suggestions** - determining whether and how to present the spelling correction
- brands that appear as spelling errors can be manually overridden, those tokens will often appear frequently
- spelling correction may also need to happen in the index

## Query classification
- iphone case -> phones/accessories/phone-cases
- classify frequent queries manually or heuristically (head queries) you can use head and torso data as training data to train a model to classify your tail (it is not ideal as they are not representative data but you can control on a sample)
- queries often not associated with a leaf node of a category taxonomy

### Manual classification

Can be effective for head queries with diminishing returns. Crowd source labelling and think about using some kind of heuristic to reduce the set of candidates for human labellers to choose from.

### exercise

```
python /workspace/search_with_machine_learning_course/utilities/categoryViewer.py --max_depth 2
```

| Query           | Manually chosen category      																			|
|-----------------|---------------------------------------------------------------------------------------------------------|
| lcd tv          | Best Buy > TV & Home Theater > TVs > LCD TVs															|
| Iphone 4s		  | Best Buy > Name Brands > Apple > Apple iPhone > Apple iPhone 4S											|
| laptop		  | Best Buy > Computers & Tablets > Laptop & Netbook Computers												|
| star wars		  | Best Buy > Home > Cartoon & Popular Characters > Star Wars Merchandise									|
| transformers	  | Best Buy > Home > Cartoon & Popular Characters > Transformers											|
| ps3			  | Best Buy > Video Games > PlayStation 3																	|
| wireless router | Best Buy > Computers & Tablets > Networking & Wireless > Routers > Wireless Routers						|
| beats			  | Best Buy > Name Brands > Club Beats > Beats Headphones > Beats by Dre									|
| thor			  | Best Buy > Movies & Music > Digital Movies																|
| hp tablet		  | Best Buy > Computers & Tablets > Tablets & iPad > Tablets OR Best Buy > Name Brands > HP > Computers	|

### Heuristic

- string matching and regular expression approaches work well for short texts such as queries

```
cd /workspace/search_with_machine_learning_course/week3
head /workspace/datasets/train.csv | cut -d , -f3 | python leavesToPaths.py --max_depth 3 
```
[Grep with regular expression](https://linuxize.com/post/regular-expressions-in-grep/)

```
grep -i -E '(laptop)?s' /workspace/datasets/train.csv | cut -d',' -f3 | python leavesToPaths.py --max_depth 3 | sort | uniq -c | sort -nr | head
```

| Category 	| Heuristic      																							|
|-----------|-----------------------------------------------|
| TVs		| -i -E '\b(tv)?s\b|\b(television)?s\b'
| Laptops	|
| Music		|



### Getting categories by mining logs

Check the top couple of categories returned, the count for further categories should drastically fall off. This works well for high frequency queries but not for less frequent ones. Manual classification like this is ok for perhaps 1000 queries, heuristics will help get the next 10,000 but scaling beyond that is where ML comes in.

- we need labeled data for training
- we need to make sure it is representative
- we need to include negative examples (category unknown)

### Hierarchy

categories often fit into a hierarchical taxonomy, often we map to the leaf category but its not always the case especially for queries.

- aggregate leaf level classifications (calculate probability of a parent by the sum of its children's probability)
- start at leaf and keep going up until we reach our threshold
- level specific classifiers

## Integrating Query classification into search
- use it to filter results
- as a boost

## Notes from Class

[queryunderstanding.com](https://queryunderstanding.com/) from Daniel Tunkelang

### Query understanding versus Ranking when does one perform better over the other?
- when historical interactions (judgments) depict better the intent than a concrete query
- QU solves the problem of setting the search scope properly. Whereas ranking truly shines when the matched candidates already properly represent the searcher's intent, and then it is matter of ordering items
- QU as having a strategy part (set the scope) and a tactical part: identify key tokens/phrases, etc. Both of these then inform not only the query you would ultimately submit, but also what ranking approaches you might choose or even if you do a search at all!
- not doing a search (cache look up) or if you know the answer (question answering system for frequently asked)
- impact ordering in the index, cutting off the search when you know you have enough results
- perhaps you may also need more than one search, ie when you do query expanding or if you have multiple intents and want to treat each one as a subproblem ie ipad could require a lookup for the device and also the accessories
- time based split, ie if someone looks up a historical figure who was recently in the news we might have one search for recent events and a second for significant results
- rule based sitting in front but we want this to stay as small as possible

### Query understanding is useful as a guardrail
- categorize and synonyms will help prevent precision loss while we still increase recall

### Project
- category hierarchy and pruning the category taxonomy will be important

### Known-item search versus exploratory search

Known search normally result in one click, tighter search results, entity recognition might also let you decide if its known-item-search, often near or exact matches whereas exploratory will have multiple clicks, a wider range of results. Instant search are good candidates for known-item-search. 

### Personalization

The strategy part of query understanding, you can classify in different ways and have multiple models, personalization aspects. Show apple apps over android apps (use prior information)

## Knowledge graph

Blend results from QA model or knowledge graph.

### Reading material

[ai for query understanding](https://dtunkelang.medium.com/ai-for-query-understanding-d8c073095fff)
[using ai to understand search intent](https://dtunkelang.medium.com/using-ai-to-understand-search-intent-1fef055e901f)
[focus on session success](https://us06st1.zoom.us/web_client/4qu8baa/html/externalLinkPage.html?ref=https://dtunkelang.medium.com/supporting-the-searchers-journey-when-and-how-568e9b68fe02)
[measuring search effectiveness](https://dtunkelang.medium.com/measuring-search-effectiveness-a320bd6bdd7a)
[untold silicon valley](https://www.untoldsiliconvalley.com/)
[knowledge graphs](https://thenoisychannel.com/2011/11/15/cikm-2011-industry-event-john-giannandrea-on-freebase-a-rosetta-stone-for-entities/)