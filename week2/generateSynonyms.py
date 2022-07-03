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