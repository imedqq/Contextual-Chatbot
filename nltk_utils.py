import nltk
#nltk.download('punkt') #package with pretrained tokenizer, download first time running
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
	return nltk.word_tokenize(sentence)

def stem(word):
	return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
	pass

a = "How long does shipping take?"
print(a)
a = tokenize(a)
print(a)
a = [stem(word) for word in a]
print(a)
