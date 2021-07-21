import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


with open('intents.json', 'r') as f:
	intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
	tag = intent['tag']
	tags.append(tag)
	for pattern in intent['patterns']:
		w = tokenize(pattern)
		# extend because w is an array, and we don't want an array of arrays in all_words
		all_words.extend(w)
		# pattern with corresponding tag
		xy.append((w, tag))

ignore_words = ['?','!','.',',']
# apply stemming, exclude ignore words with list comprehension
all_words = [stem(w) for w in all_words if w not in ignore_words]
# sort words and use set a neat trick to remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(tags)

# Bag of Words
# 	 In Linear Algebra, it is extremely common to use capital Latin letters for matrices and lowercase Latin letters for vectors
# Usually X is a matrix of data values with multiple feature variables, having one column per feature variable
# and y is a vector of data values
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
	bag = bag_of_words(pattern_sentence, all_words)
	X_train.append(bag)

	label = tags.index(tag)
	y_train.append(label) # CrossEntropyLoss

X_train = np.array(X_train)
y_train = np.array(y_train)

# create pytorch dataset from X_train and y_train training data
class ChatDataset(Dataset):
	def __init__(self):
		self.n_samples = len(X_train)
		self.x_data = X_train
		self.y_data = y_train

	#access dataset with index
	#dataset[idx]
	def __getitem__(self, index):
		return self.x_data[idx], self.y_data[idx]

	def __len__(self):
		return self.n_samples

	# Hyperparamaters
	batch_size = 8

	dataset = ChatDataset()
	train_loader = DataLoader(dataset=dataset, batch_size=batch_size. shuffle=True, num_workers=2) #or 0

	