# Contextual Chatbot 

Chatbot using PyTorch with NLTK(Natural Language Toolkit)

Reads from json file and compares user inputs with specifically trained questions in different categories to determine the correct category it most closely falls into, then replies with a random answer from a number of preset responses according to the category of question

Currently set up as a chat bot for an online shop that sells coffee and tea with minimal data trained with


## Run
```console
python chat.py
```
Type quit to exit

## Install

Create new environment in Anaconda
```console
conda create --name pytorch python=3.8
```

In Anaconda console
```console
conda activate pytorch
pip install nltk

```

Download nltk tokenizer package
Uncomment comment in nltk_utils.py
```console
python nltk_utils.py
```

## Training
```console
python train.py
```
Data saved in data.pth

## Notes
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

Feed Forward Neural Network
- Two layers
- Takes bag of words input, pass through layer with number of patterns as input size, then a hidden layer, then the output size must be number of classes, then apply softmax probability to each classes

Bag of words

Tokenization: Splitting a string into meaningful units (e.g. words, punctuation characters, numbers)
"what would you do with $1,000,000?"
->["what","would","you","do","with","$","1,000,000","?"]

Stemming: Generate the root form of the words. Crude heuristic that chops off the ends of words
"organize","organizes","organizing"
->["organ","organ","organ"]
