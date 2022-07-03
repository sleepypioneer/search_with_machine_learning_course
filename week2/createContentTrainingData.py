import argparse
import glob
import multiprocessing
import os
import re
import random
from typing import List, Tuple
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from nltk.stem import SnowballStemmer


stemmer = SnowballStemmer("english")

def transform_name(product_name):
    product_name = re.sub(r"'\W+'mg", "", product_name)
    product_name = product_name.lower().strip()
    return stemmer.stem(product_name)

# Directory for product data
directory = r'/workspace/datasets/product_data/products/'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--label", default="id", help="id is default and needed for downsteam use, but name is helpful for debugging")

# Consuming all of the product data, even excluding music and movies,
# takes a few minutes. We can speed that up by taking a representative
# random sample.
general.add_argument("--sample_rate", default=1.0, type=float, help="The rate at which to sample input (default is 1.0)")

# IMPLEMENT: Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=50, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
# IMPLEMENT:  Track the number of items in each category and only output if above the min
min_products = args.min_products
sample_rate = args.sample_rate
names_as_labels = False
if args.label == 'name':
    names_as_labels = True


def _label_filename(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    labels = []
    for child in root:
        if random.random() > sample_rate:
            continue
        # Check to make sure category name is valid and not in music or movies
        if (child.find('name') is not None and child.find('name').text is not None and
            child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
            child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None and
            child.find('categoryPath')[0][0].text == 'cat00000' and
            child.find('categoryPath')[1][0].text != 'abcat0600000'):
              # Choose last element in categoryPath as the leaf categoryId or name
              if names_as_labels:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][1].text.replace(' ', '_')
              else:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
              # Replace newline chars with spaces so fastText doesn't complain
              name = child.find('name').text.replace('\n', ' ')
              labels.append((cat, transform_name(name)))
    return labels

if __name__ == '__main__':
    files = glob.glob(f'{directory}/*.xml')

    with multiprocessing.Pool() as p:
        labels_df = pd.DataFrame()
        all_labels = tqdm(p.imap_unordered(_label_filename, files), total=len(files))
        for label_list in all_labels:
            labels_df = labels_df.append(label_list)
        
        labels_df = labels_df.rename(columns={0: "category", 1 : "product_name"})
        categories_to_keep = labels_df["category"].value_counts().loc[lambda x : x > min_products].index.to_list()

        pruned_labels_df = labels_df.loc[labels_df["category"].isin(categories_to_keep)]

        pruned_labels_df["output_string"] = pruned_labels_df[["category", "product_name"]].apply(lambda row: f'__label__{row[0]} {row[1]}', axis=1)

        print(f"Writing {len(pruned_labels_df)} results to {output_file}")
        with open(output_file, 'w') as output:
            for entry in pruned_labels_df["output_string"]:
                output.write(f'{entry}\n')