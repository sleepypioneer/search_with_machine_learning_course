# Project week One - Building a Learning to Rank model & evaluating it

1. Implementing End to End LTR for our Dataset 
Level 
2. Exploring Features and Click Models
3. Relevance Judgments on LTR

**⚠️ synthetic impressions dataset**
created via:
1. Running our queries through a “baseline retrieval” that will collect a full set of impressions (e.g. the results a user would have seen if they were using this engine) and overlaying them with our clicks
2. Inferring the ranking based off the distribution of clicks and relying on position bias in our favor. 


first run (without implementing logging)

```sh
Meta:
Model name: ltr_model, Store Name: week1, Index: bbuy_products, Precision: 10 

Zero results queries: {'simple': [], 'ltr_simple': [], 'hand_tuned': [], 'ltr_hand_tuned': []}

Queries not seen during training: [6]
                 query
0  amplificador pioner
1           mint floor
2       Ipod touch mp3
3  kipsch speakers 2.1
4       cars 2 blu ray
5    defender iphone 4


Simple MRR is 0.310
LTR Simple MRR is 0.310
Hand tuned MRR is 0.391
LTR Hand Tuned MRR is 0.391

Simple p@10 is 0.110
LTR simple p@10 is 0.110
Hand tuned p@10 is 0.160
LTR hand tuned p@10 is 0.160
Simple better: 0        LTR_Simple Better: 0    Equal: 971
HT better: 0    LTR_HT Better: 0        Equal: 1129
```

Second run after completing steps 1a - 4e

```sh
Meta:
Model name: ltr_model, Store Name: week1, Index: bbuy_products, Precision: 10 

Zero results queries: {'simple': [], 'ltr_simple': [], 'hand_tuned': [], 'ltr_hand_tuned': []}

Queries not seen during training: [13]
                                  query
0           Dynex Digital picture frame
1                     Apple iPad 2 32gb
2               2371074 2126339 2126108
3                       Travel chargers
4                   vhs tape converters
5                                PacMan
6                        Garmon 1350lmt
7                             iPhone 4s
8                            IPod radio
9                ihome speaker for ipad
10            External hard drive cases
11                            Microfono
12  Making of the dark side of the moon


Simple MRR is 0.282
LTR Simple MRR is 0.281
Hand tuned MRR is 0.370
LTR Hand Tuned MRR is 0.350

Simple p@10 is 0.105
LTR simple p@10 is 0.105
Hand tuned p@10 is 0.181
LTR hand tuned p@10 is 0.173
Simple better: 62       LTR_Simple Better: 24   Equal: 1254
HT better: 309  LTR_HT Better: 484      Equal: 659
```

python week1/utilities/build_ltr.py --xgb_test /workspace/ltr_output/test.csv --train_file /workspace/ltr_output/train.csv --output_dir /workspace/ltr_output --xgb_test_num_queries 200 **--xgb_main_query_weight 1** **--xgb_rescore_query_weight 1000**  && python week1/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output 

```sh
Simple MRR is 0.294
LTR Simple MRR is 0.290
Hand tuned MRR is 0.422
LTR Hand Tuned MRR is 0.388

Simple p@10 is 0.119
LTR simple p@10 is 0.112
Hand tuned p@10 is 0.171
LTR hand tuned p@10 is 0.151
Simple better: 427      LTR_Simple Better: 233  Equal: 1586
HT better: 1074 LTR_HT Better: 815      Equal: 585
```

python week1/utilities/build_ltr.py --xgb_test /workspace/ltr_output/test.csv --train_file /workspace/ltr_output/train.csv --output_dir /workspace/ltr_output --xgb_test_num_queries 200 **--xgb_main_query_weight 1000** **--xgb_rescore_query_weight 1**  && python week1/utilities/build_ltr.py --analyze --output_dir /workspace/ltr_output

```sh
Simple MRR is 0.299
LTR Simple MRR is 0.299
Hand tuned MRR is 0.427
LTR Hand Tuned MRR is 0.427

Simple p@10 is 0.100
LTR simple p@10 is 0.100
Hand tuned p@10 is 0.191
LTR hand tuned p@10 is 0.191
Simple better: 0        LTR_Simple Better: 0    Equal: 2479
HT better: 118  LTR_HT Better: 202      Equal: 2434
```

## Level 2 Click Models

Heuristic click model
```python
data_frame["grade"] = (data_frame["clicks"] / (data_frame["num_impressions"] + prior)).fillna(0).apply(lambda x: step(x))
if downsample:
	#print("Size pre-downsample: %s\nVal Counts: %s\n" % (len(data_frame), data_frame['grade'].value_counts()))
	data_frame = down_sample_buckets(data_frame)
	#print("Size post-downsample: %s\nVal Counts: %s\n" % (len(data_frame), data_frame['grade'].value_counts()))
```

- We are calculating a modified CTR as clicks / (num impressions + prior)
- We are passing in a prior and adding it to the denominator to help reduce the impact of tail queries
- We are applying the step function which maps all CTRs into 4 buckets based on their score.  *Note, the step function ranges were picked arbitrarily in an attempt to balance out our grades, but it doesn’t work all that well if the prior changes.*


Using Quantiles Click model

`./ltr-end-to-end.sh -y -m 0 -c quantiles`

```python
ata_frame["grade"] = pd.qcut((data_frame["clicks"] / (data_frame["num_impressions"] + prior)).fillna(0), quantiles, labels=False) / quantiles
```

```sh
CTR Quantiles click model
Grade distribution:  0.6    877
0.0    877
0.1    877
0.5    876
0.8    876
0.9    876
0.4    876
0.3    876
0.2    875
0.7    875

Simple MRR is 0.249
LTR Simple MRR is 0.164
Hand tuned MRR is 0.367
LTR Hand Tuned MRR is 0.168

Simple p@10 is 0.097
LTR simple p@10 is 0.068
Hand tuned p@10 is 0.144
LTR hand tuned p@10 is 0.064
Simple better: 742      LTR_Simple Better: 606  Equal: 8
HT better: 833  LTR_HT Better: 550      Equal: 7
```

After adding more features [match_phrase, customerReviewAverage, customerReviewCount, artistName_match, shortDescription_match, longDescription_match] and running with quantiles model.

```sh
Simple MRR is 0.311
LTR Simple MRR is 0.313
Hand tuned MRR is 0.427
LTR Hand Tuned MRR is 0.335

Simple p@10 is 0.095
LTR simple p@10 is 0.153
Hand tuned p@10 is 0.182
LTR hand tuned p@10 is 0.158
Simple better: 344      LTR_Simple Better: 437  Equal: 6
HT better: 488  LTR_HT Better: 489      Equal: 11
```

Added salesRankShortTerm feature:

```sh

```


## Project Assessment

1. Do you understand the steps involved in creating and deploying an LTR model?  Name them and describe what each step does in your own words.

- initialize the storage in Opensource
- populate the feature store with the feature set, we can adjust this in our config file for the feature set. 
- collect judgments (this can be implicit or explicit, in our case we used click through information to infer user intent)
- create a training data set by logging feature scores
- train & test the model
- analyze your model and make adjustments this probably means going right back to the feature set and adjusting the values there.
- once happy you can deploy the model by POSTing to the `createmodel` endpoint.


2. What is a feature and features?

- A feature is a specified attribute of a item such as price, size, color etc
- A feature set is a group of features normally represented in an array.

3. What is the difference between precision and recall?

- Precision is calculated by taking the number of true positives divided by the sum of the true positives and false positives, we value it when we care more that the results are only from the desired class

- Recall is calculated by taking the number of true positives divided by the sum of true positives and false negatives, we value it when we want to collect more results of our class but ware willing to risk some of them will not be the of the right class.

4. What are some of the traps associated with using click data in your model?

Clicks don't always represent a users intent, dwell time might be an interesting metric for example when looking at a result such as an info box which might not result in a click. Clicks are also biased by rank, some results won't suffice to the user and therefore won't be clicked.

5. What are some of the ways we are faking our data and how would you prevent that in your application?

We are faking impressions, we would prevent this by ensuring we log out more information (such as position). It's uncommon to find data like this that is open which is why this was required. We also don't know where on the page results appeared so we are inferring this from if they were clicked. We might also want different data such as dwell time because as mentioned before using only CTR can not be sufficient.

6. What is target leakage and why is it a bad thing?

Target leakage happens when your training dataset includes information that would not be available at the time of prediction, the model will therefore be unrealistically accurate for the training data (overfitting).

7. When can using prior history cause problems in search and LTR?

When things change (ie a new product version is launched) prior click data will not help.

8. Submit your project along with your best MRR scores