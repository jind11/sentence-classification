# sentence-classification
Implementation of sentence classification using CNN, CNN-RNN, fasttext, etc. References to these models are listed here:

| Model        | References | 
| ------------- |:-------------:|
| fasttext      | https://arxiv.org/abs/1607.01759 |
| cnn      |   https://arxiv.org/abs/1408.5882   |
| cnn-rnn |  https://www.aclweb.org/anthology/C/C16/C16-1229.pdf  |

## Dependencies

This code is written in python and the deep learning library is keras. To use it you will need:

* Python 2.7
* keras
* theano (this can be avoided by choosing tensorflow as the backend)
* numpy
* tensorflow (for vocabulary indexing)
* gensim (for reading word2vec bin file)
* cPickle (for text data storage and loading)

## Getting started

1. We use [Movie Review data from Rotten Tomatoes](http://www.cs.cornell.edu/people/pabo/movie-review-data/) as the demo dataset. For your convenience, the dataset is already saved in the data directory.

2. To boost performance for CNN and CNN-RNN model, pre-trained word embeddings can be used as the initialization of word embeddings. To use it, you need to download the [Google word2vec binary file](https://code.google.com/p/word2vec/). For your convenience, a bash file named 'word2vec_download.sh' has been written well in the code directory to download the bin file, save it in the data directory and unzip it. To use the script, just type in 'bash word2vec_download.sh' in the code directory in the bash terminal.

3. To preprocess the raw data, the 'preprocess_data_and_labels_MR' function in the script 'util.py' is for CNN/CNN-RNN model, which indexes the whole corpus (create a vocabulary for the corpus and convert each token to an index based on this vocabulary) and trims the word2vecc embeddings based on the vocabulary obtained in the last step. On the other hand, the 'preprocess_data_and_labels_MR_fasttext' function in the script 'util.py' is for fasttext model, which indexes the whole corpus and adds n-grams indices such as bigrams and trigrams.

## Running the models

This code can be run in either cpu or gpu devices. For instance of theano backend, to run the code in cpu device, just type in:
```
python crnn.py
python cnn.py
python fasttext.py
```
To run the code in gpu device, just type in:
```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python crnn.py 
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python fasttext.py 
```

## Running results
The test accuracy resultes listed here are average results of 10 folds cross validation. The training/development/test split ratio is 0.8/0.1/0.1.

| Model        | Test Accuracy (%)| 
| ------------- |:-------------:|
| fasttext      | 78.1 |
| cnn      |   80.1    |
| cnn-rnn | 82.2      |

To be noted, the fasttext model does not use pre-trained word embeddings and instead uses random initialized word embeddings (bigrams and trigrams do not have pre-trained word embeddings), which should be the main reason of relatively lower test accuracy. To demonstrate this statement, we specially trained the cnn model with random initialized word embeddings and list the comparison results here:

| Model        | Test Accuracy (%)| Training time for one epoch (s) | Typical needed epochs |
| ------------- |:-------------:| :----------: | :------------: |
| cnn-random     | 77.8 | 60 | 5 |
| fasttext      |   78.1    | 10 | 5 |

Both tests listed above are run in the i5 cpu. In terms of accuracy, the fasttext model has comparable capability to the CNN model with random initializations of word embeddings, however, the fasttext model is much faster in terms of training time. The boosted training speed lies in the fact that fasttext does not have convolutional layer, which takes time for computation.

## To be continued

Implementations of more models will be updated later. If you have any suggestions or questions, feel free to post an issue here.
