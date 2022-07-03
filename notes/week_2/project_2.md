# Project 2


## Answers

1.
a) My final scores were:
```
N       10000
P@1     0.967
R@1     0.967
```
b) `-epoch 25 -lr 1`
c) Name transformation done in [createContentTrainingData.py](../../week_2/createContentTrainingData.py)

```python
import re
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

def transform_name(product_name):
	# normalization is the same as what was done before
    product_name = re.sub(r"'\W+'mg", "", product_name)
    product_name = product_name.lower().strip()
    return stemmer.stem(product_name)
```
d) The code for this is also in [createContentTrainingData.py](../../week_2/createContentTrainingData.py)
e) TODO

2.
a)
Query `iphone`
iphone® 0.917707
3gs 0.827327
4s 0.785832
3g 0.785089

Query `earbuds`
earbud 0.995359
bud 0.828973
headphones 0.816021
ear 0.814233
on-ear 0.778627
headphon 0.776735
over-the-ear 0.753068

Query `washing machine`
7-string 0.815901
viking 0.813331
washburn 0.791989
6-string 0.782053
trans 0.768159
guitars 0.765206



b) model='skipgram',epoch=25, minCount=20
c) I used the same as previous

3.
a) I used the same as previous
b) 0.75
c)

| Query     | No. results (name)| No. results (name.synonym)|
|-----------|-------------------|---------------------------|
| earbuds   | 1205
| nespresso | 8
| dslr      | 2837

4. TODO


## Notes from kick off

- smaller index
- generate training data
- running fast text
	- classification: supervised
	- synonyms: skipgrams, nearest neighbor
- integrate synonyms into search


### Document catogorisation

- get product name and category
- put into key value keys that fastText can process
- for class we are running training and testing on two different 10% samples (10,000) for speed reasons
	- we need to shuffle the data to remove artifacts from ordering
- only keep entries that are associated with a label assigned to at least 500 products. (pandas df or dictionary)
- aim for precision above 90%
- because we throw away categories we don't have much data for. We could therefore look up the hierarchy (read category XML)

### Synonyms

- product titles with labels removed
- normalization required
- build synonym set
- integrate with search
	- extract most frequent words
	- run nn and keeps pairs above a certain threshold
	- integrate to analyzer in Opensearch as synonym token filter
- [Synonyms and search](https://dtunkelang.medium.com/real-talk-about-synonyms-and-search-bb5cf41a8741)
- FastText and similar approaches return highly correlated terms but not true synonyms but these could be a filter. Query rewrites could also be taken as a synonym but it's worth having some human layer to check.
- Knowledge graphs
	- very expressive (complex)
	- quality can be patchy, one thing might have a very complete data coverage while another might not (especially crowd sourced)
	- is this complexity worth the return
- recall problem with synonyms, the synonym is frequent but returning very large result sets, or low CTR might be an indicated that the synonym is having a negative effect

The user uses a word that wouldn't have matched any results but not match a bunch of results, so they appear highly in the queries but not in the logs (as they lead to small or empty result sets) this is where they have the most impact - note this could be good or bad.

## Lemmatization versus stemming

Lemmatization is safer, same semantic form while stemming is heuristic and follows rules like removing endings, it can however work for words which are unknown. More [here](https://queryunderstanding.com/stemming-and-lemmatization-6c086742fe45)


LTR pipeline:
- build a feature store in opensearch for our ranking algorithm
- add our model (XGBoost)
- use logging to evaluate our predictions for queries
- give weights to our features to improve MMR with the model

Cold start problem:
- we don't have enough logs to mine (ie new index or low volume)
- BM25 is very effective
- you need judgments that generalize to most query-document pairs



Conferences SpecialInterestGroupInformationRetrieval, KnowledgeDiscoveryD
[reinforcement learning assisted search](https://us06st1.zoom.us/web_client/4qu8baa/html/externalLinkPage.html?ref=https://medium.com/sajari/reinforcement-learning-assisted-search-ranking-a594cdc36c29)



## Level 1

Create training data with labels:

```
python week2/createContentTrainingData.py --output /workspace/datasets/fasttext/labeled_products.txt
 ```

Shuffel training data:

```
shuf /workspace/datasets/fasttext/labeled_products.txt > /workspace/datasets/fasttext/shuffled_labeled_products.txt
```

Find length of data:

```
wc -l /workspace/datasets/fasttext/shuffled_labeled_products.txt
```

Take first 10,000 entries for training:

```
( head -10000 /workspace/datasets/fasttext/shuffled_labeled_products.txt) > /workspace/datasets/fasttext/training_data.txt
```

Take second 10,000 entries for testing:

```
sed -n 10001,20000p  /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/test_data.txt
```

Also useful command, this is to remove last line:

```
 sed -i '$d'  /workspace/datasets/fasttext/test_data.txt
```

Train our model on the training data:

```
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/training_data.txt -output model_bb

Read 0M words
Number of words:  11159
Number of labels: 1365
Progress: 100.0% words/sec/thread:     687 lr:  0.000000 avg.loss: 13.346553 ETA:   0h 0m 0s
```

Get predictions:

```
~/fastText-0.9.2/fasttext predict model_bb.bin -
```

Evaluate model with test data:

```
 ~/fastText-0.9.2/fasttext test model_bb.bin /workspace/datasets/fasttext/test_data.txt

N       9663
P@1     0.108
R@1     0.108
 ```

Re-evaluate with 5 values:

```
 ~/fastText-0.9.2/fasttext test  model_bb.bin /workspace/datasets/fasttext/test_data.txt 5
N       9663
P@5     0.0403
R@5     0.201
```

Re-evaluate with 10 values:

```
~/fastText-0.9.2/fasttext test  model_bb.bin /workspace/datasets/fasttext/test_data.txt 10
N       9663
P@10    0.0241
R@10    0.241
```

Retrain with `-epoch 25 -lr 1.0` to create model_bb_2:

```
Read 0M words
Number of words:  11159
Number of labels: 1365
Progress: 100.0% words/sec/thread:     224 lr:  0.000000 avg.loss:  1.020993 ETA:   0h 0m 0s
```

Evaluation of model_bb_2:

```
N       9663
P@1     0.62
R@1     0.62
```

Retrain with `-wordNgrams 2` to create model_bb_3:

```
Read 0M words
Number of words:  11159
Number of labels: 1365
Progress: 100.0% words/sec/thread:     229 lr:  0.000000 avg.loss:  1.281904 ETA:   0h 0m 0s
```

Evaluation of model_bb_3:

```
N       9663
P@1     0.609
R@1     0.609
```

Normalising the training data:

```
cat /workspace/datasets/fasttext/training_data.txt |sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]_]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_training_lite.txt
```
- remove all non-alphanumeric other than underscore
- convert to lowercase
- trim white spaces

**Be sure to normalise the test data in the same way**

Retrain with normazlied data for model_bb_4:

```
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/normalized_training_lite.txt -output model_bb_4 -epoch 25 -lr 1.0
```

Evaluate with normalized test data:

```
N       9663
P@1     0.62
R@1     0.62
```

Evaluation after adding stemmer and putting normalization into python script (model_bb_5):

```
N       9614
P@1     0.63
R@1     0.63
```

```python
import re
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer("english")

def transform_name(product_name):
	# normalization is the same as what was done before
    product_name = re.sub(r"'\W+'mg", "", product_name)
    product_name = product_name.lower().strip()
    return stemmer.stem(product_name)
```

**You generally don’t want to train a classifier with fewer than 50 examples.**

Only take categories with 500 products:

```
python week2/createContentTrainingData.py --output /workspace/datasets/fasttext/labeled_products.txt --min_products 500
```

Evaluation of model model_bb_6 trained and tested with only products belonging to a category that has high representation:

```
~/fastText-0.9.2/fasttext test  model_bb_6.bin /workspace/datasets/fasttext/test_data.txt
N       10000
P@1     0.967
R@1     0.967
```

**Note this has cost us coverage**

A more robost approach is to lerveage the hierarchical nature of our taxononmy and roll up infrequestly used labels to their parent or other ancestor categories.

Prints out the full path of every category path:

```python
categoriesFilename = '/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'
tree = ET.parse(categoriesFilename)
root = tree.getroot()

for child in root:
    catPath = child.find('path')
    idArray = [cat.find('id').text for cat in catPath]
    print(idArray)
```

## Level 2: Derive Synonyms from Content

Get unlabeled list of product names:

```
cut -d' ' -f2- /workspace/datasets/fasttext/shuffled_labeled_products.txt > /workspace/datasets/fasttext/titles.txt
```

Feed this to skipgram

```
~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/title_model
```

Use nearest neighbour method on this model to get a sense of the quality of synonyms it generates:

```
~/fastText-0.9.2/fasttext nn /workspace/datasets/fasttext/title_model.bin
```

Let's do that again but normalise the data first:

```
cat /workspace/datasets/fasttext/titles.txt | sed -e "s/\([.\!?,'/()]\)/ \1 /g" | tr "[:upper:]" "[:lower:]" | sed "s/[^[:alnum:]]/ /g" | tr -s ' ' > /workspace/datasets/fasttext/normalized_titles.txt
```

We also increase the number of epochs to 25 and set the `-minCount` to 20 to remove rare words.

```
~/fastText-0.9.2/fasttext skipgram -input /workspace/datasets/fasttext/titles.txt -output /workspace/datasets/fasttext/normalized_title_model_2 -minCount 20 -epoch 25
```

Extract the top words in our product titles:

```
cat /workspace/datasets/fasttext/normalized_titles.txt | tr " " "\n" | grep "...." | sort | uniq -c | sort -nr | head -1000 | grep -oE '[^ ]+$' > /workspace/datasets/fasttext/top_words.txt
```

The above does the following:
- Replace each space with a newline, so we get one word per line.
- Only keep words containing at least 4 characters.
- Sort the words in alphabetical order.
- Deduplicate the words and keep the counts, yielding a 2-columns file where each line is a count followed by the word.
- Sort the count-word pairs in descending order of count.
- Keep only the top 1,000 entries, i.e., the 1,000 most frequently occurring words.
- Remove the counts so we only output the words.
- Output the result of this process to /workspace/datasets/fasttext/top_words.txt.

Generate Synonyms for our top words and save them in a CSV:

```python
import argparse
import fasttext

model = fasttext.load_model("/workspace/datasets/fasttext/normalized_title_model_2.bin")

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--top_words", default="/workspace/datasets/fasttext/top_words.txt",  help="The file containing the top words")
general.add_argument("--output", default="/workspace/datasets/fasttext/synonyms.csv", help="the file to output to")
general.add_argument("--threshold", default=0.75, help="the threshold on which to include a synoynm")


args = parser.parse_args()
top_words_file = args.top_words
output_file = args.output
threshold = args.threshold

if __name__ == '__main__':
	words_file = open(top_words_file)
	words = words_file.readlines()
	entries = []

	for word in words:
		word = word.replace('\n','')
		entry = [word]
		synonyms = model.get_nearest_neighbors(word)
		for synonym in synonyms:
			# if the score is above our threshold
			if float(synonym[0]) > threshold:
				entry.append(synonym[1])
		entries.append(entry)

	with open(output_file, 'w') as output:
		for entry in entries:
			output.write(f'{", ".join(entry)}\n')

```

```
docker cp /workspace/datasets/fasttext/synonyms.csv opensearch-node1:/usr/share/opensearch/config/synonyms.csv
```

Integrate the synonym to the indexing process (you could do this at query time, which is the preferred way) using the [synonym token filter](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-synonym-tokenfilter.html)



#### Reading from Coding party:

[Function score query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-function-score-query.html)
[Built in analyzer](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-analyzers.html)