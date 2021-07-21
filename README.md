# Contextual Chatbox 

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

NLP Basics

Feed Forward Neural Net
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

