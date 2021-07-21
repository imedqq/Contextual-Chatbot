import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

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

	# support indexing such that dataset[i] can be used to get i-th sample
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	# we can call len(dataset) to return the size
	def __len__(self):
		return self.n_samples

# Hyperparamaters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0]) # use either len of all_words or first bagofwords because they all have same size
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) # or 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(num_epochs):
	for (words, labels) in train_loader:
		words = words.to(device)
		labels = labels.to(dtype=torch.long).to(device)

		# forward
		outputs = model(words)
		# if y would be one-hot, we must apply
		# labels = torch.max(labels, 1)[1]
		loss = criterion(outputs, labels)

		# backward and optimizer step, empty gradients first
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if (epoch +1) % 100 == 0: # every 100 step
		print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss{loss.item():.4f}')