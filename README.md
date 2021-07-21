# Contextual Chatbot 

Chatbot using PyTorch with NLTK(Natural Language Toolkit)

Our NLP Preprocessing Pipeline
"Is anyone there?"
-> tokenize
["Is","anyone","there","?"]
-> lower + stem
["is","anyon","there","?"]
-> exclude punctuation characters
["is","anyon","there"]
-> bag of words
[0,0,0,1,0,1,0,1]

1) Theory + NLP concepts (stemming, tokenization, bag of words)
2) Create training data
3) PyTorch model and training
4) Save/load model and implement the chat

Feed Forward Neural Net
- Two layers
- Takes bag of words input, pass through layer with number of patterns as input size, then a hidden layer, then the output size must be number of classes, then apply softmax probability to each classes

Bag of words

Tokenization: Splitting a string into meaningful units (e.g. words, punctuation characters, numbers)
"what would you do with $1,000,000?"
->["what","would","you","do","with","$","1,000,000","?"]

Stemming: Generate the root form of the words. Crude heuristic that chops off the ends of words
"organize","organizes","organizing"
->["organ","organ","organ"]


## Install and run

Create new environment in Anaconda
```console
conda create --name pytorch python=3.6
```

In Anaconda console
```console
conda activate pytorch
pip install nltk

```

download nltk tokenizer package
uncomment comment in nltk_utils.py
```console
python nltk_utils.py
```
