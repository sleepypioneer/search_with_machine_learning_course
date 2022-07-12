# Project 3

Classifier than classifies a query to one or more categories, with the plan to then use that to boost the query.

Use same training data as before but only focus on categories of the results.

## Notes from kickoff

Good to have easily accessible dashboards for queries

Continuous learning -> fallible to trends or user spikes

Google has multiple indices ready to go, including some more basic indices that they can roll back to.

## Level 1 Query classification

### Prune Category taxonomy

- **if** num of associated queries to a category **<** minimum **then** roll up to parent category

Create data set with labels:

```sh
python week3/create_labeled_queries.py --min_queries 10000 --output /workspace/datasets/labeled_query_data.txt
```

Shuffel training data:

```sh
shuf /workspace/datasets/labeled_query_data.txt > /workspace/datasets/fasttext/labeled_query_data.txt
```

Find length of data:

```sh
wc -l /workspace/datasets/fasttext/labeled_query_data.txt
> 1850373
```

Take first 50,000 entries for training:

```sh
( head -50000 /workspace/datasets/fasttext/labeled_query_data.txt) > /workspace/datasets/fasttext/training_data.txt
``` 

Take second 10,000 entries for testing:

```sh
sed -n 50001,60000p  /workspace/datasets/fasttext/labeled_query_data.txt > /workspace/datasets/fasttext/test_data.txt
```

Train our model on the training data:


```sh
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/training_data.txt -output model_bb_8

Number of words:  7910
Number of labels: 387
Progress: 100.0% words/sec/thread:     836 lr:  0.000000 avg.loss:  4.200980 ETA:   0h 0m 0s
```

Evaluate model with test data with recall 1:

```sh
~/fastText-0.9.2/fasttext test model_bb_8.bin /workspace/datasets/fasttext/test_data.txt 1

N       10000
P@1     0.477
R@1     0.477
```

Re-evaluate with recall 3:

```sh
~/fastText-0.9.2/fasttext test  model_bb_8.bin /workspace/datasets/fasttext/test_data.txt 3

N       10000
P@3     0.211
R@3     0.634
```


Re-evaluate with recall 5:

```sh
~/fastText-0.9.2/fasttext test  model_bb_8.bin /workspace/datasets/fasttext/test_data.txt 5

N       10000
P@5     0.14
R@5     0.701
```

Increase the number of epochs and change the learning rate:

```sh
~/fastText-0.9.2/fasttext supervised -input /workspace/datasets/fasttext/training_data.txt -output model_bb_9 -epoch 15 -lr 0.5
```

Evaluate model with test data with recall 1:

```sh
N       10000
P@1     0.519
R@1     0.519
```

Re-evaluate with recall 3:

```sh
N       10000
P@3     0.234
R@3     0.701
```


Re-evaluate with recall 5:

```sh
N       10000
P@5     0.153
R@5     0.766
```


### Model_bb_10

For a minimum number of queries per category of 10,000 (69 categories) and recall of 5

```sh
N       10000
P@5     0.167
R@5     0.834
```

### Model_bb_11

For a minimum number of queries per category of 10,000 (69 categories) with epoch 15 and learning rate of 0.5 and recall of 5

```sh
N       10000
P@5     0.167
R@5     0.834
```
and recall 1

```sh
N       10000
P@1     0.59
R@1     0.59
```

### Model_bb_12

For a minimum number of queries per category of 10,000 (69 categories) with epoch 10 and learning rate of 0.5 on training set of 100,000 (previous used 50,000) and recall of 5

```sh
N       10000
P@5     0.172
R@5     0.861
```
and recall 1

```sh
N       10000
P@1     0.609
R@1     0.609
```

## Level 2 Integrating query Classification with Search

Work done inside [query.py](../../utilities/query.py)

## Assessment

1a)  For a minimum number of queries per category of 1,000 there were 387 unique categories, for 10,000 only 69 categories remain
b) my top scores were for model_bb_12 (For a minimum number of queries per category of 10,000 (69 categories) with epoch 10 and learning rate of 0.5 on training set of 100,000 (previous used 50,000) and recall of 5)


```sh
N       10000
P@5     0.172
R@5     0.861
```

second was model_11 which also had a minimum number of queries per category of 10,000 (69 categories):

```sh
N       10000
P@5     0.167
R@5     0.834
```

2a)
b) My top sco