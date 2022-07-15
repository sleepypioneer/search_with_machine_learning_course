import argparse
import os
import re

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()


def normalize_query(query):
    query = re.sub(r"'\W+'mg", "", query)
    query = query.lower().strip()
    # stemming should happen at token level
    tokens = query.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/labeled_query_data.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])


def _get_parent_cat(cat):
    return parents_df.loc[parents_df["category"] == cat]["category"].values[0]

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

print("Normalising queries")
# LEVEL 1: normalise queries
df["query"] = df["query"].apply(normalize_query)

# LEVEL 1: Roll up categories to ancestors to satisfy the minimum number of queries per category.
category_counts = df["category"].value_counts().rename_axis("category").reset_index(name="count")
merged_df = df.merge(category_counts, how="left", on="category").merge(parents_df, how="left", on="category")

num_subcategories_under_threshold = len(category_counts[category_counts["count"] < min_queries])
print(f"Subcategories under threshold: {num_subcategories_under_threshold}")

while num_subcategories_under_threshold > 0:
    merged_df.loc[merged_df["count"] < min_queries, "category"] = merged_df["parent"]
    df = merged_df[["category", "query"]]
    df = df[df["category"].isin(categories)]
    category_counts = df["category"].value_counts().rename_axis("category").reset_index(name="count")
    merged_df = df.merge(category_counts, how="left", on="category").merge(parents_df, how="left", on="category")
    num_subcategories_under_threshold = len(category_counts[category_counts["count"] < min_queries])
    print(f"Subcategories under threshold: {num_subcategories_under_threshold}")

print(f"Number of categories in final data: {len(category_counts)}")

# Create labels in fastText format.
df['label'] = '__label__' + df['category']
# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
print(f"Training data written to {output_file_name}")
