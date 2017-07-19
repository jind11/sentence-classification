# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import time
import logging
import os.path
from os.path import join as pjoin

import numpy as np
from keras import regularizers
from keras.models import Model
from keras.layers import Dropout, Embedding
from keras.layers.core import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D, Input
from keras.layers.recurrent import GRU
from keras.layers.merge import concatenate
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam, Adadelta, SGD
from keras.constraints import maxnorm
from keras.models import model_from_json

from util import Progbar, create_batches, load_data_MR, load_embeddings

logging.basicConfig(level=logging.INFO)

"""
Define all parameters

"""
# Model Hyperparameters
output_size = 1 # The output size of your model.
num_filters = 200 # Number of filters per filter size
embedding_size = 300 # Dimensionality of word embeddings
filter_sizes = [4, 5] # A list of filter sizes
max_pool_size = 2 # The max pooling size for cnn layer.
rnn_hidden_size = 100 # Hidden layer size for rnn layer.
mlp_hidden_size = 400 # Hidden layer size for mlp layer.
embedding_dropout = .5 # Dropout for word embeddings.
cnn_dropout = .15 # Dropout for cnn layer.
mlp_dropout = .1 # Dropout for full connected hidden layer.

# Data loading and storing params
model_name = 'cnn_rnn' 
base_train_dir = "../train/" + model_name # training results store directory
data_path =  "../data" # data retrieving directory
embedding_path = "../data/word2vec.trimmed.npz" # word vector embeddings retrieving directory

# Training parameters
base_lr = .99 # Initial learning rate.
decay_rate = .0 # Decay ratio.
batch_size = 32 # Batch size to use during training and testing.
n_epochs = 130 # Number of epochs to train.
max_norm = 3.0 # L2 constraint on the weight vector of full connected hidden layer.
n_folds = 10 # Total number of folds, N fold cross validation


def setup_model(embeddings, seq_len, vocab_size):

    # Add input
    inputs = Input(shape=(seq_len, ), dtype='int32', name='inputs')

    # Add word vector embeddings
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                        weights=[embeddings], input_length=seq_len, 
                        name='embedding', trainable=True)(inputs)

    # Add dropout to the embeddings
    embedding_drop = Dropout(embedding_dropout)(embedding)

    # Add convolution layer
    conv4 = Conv1D(filters=num_filters,
                    kernel_size=filter_sizes[0],
                    padding='valid',
                    activation='relu',
                    name='conv4')(embedding_drop)

    maxConv4 = MaxPooling1D(pool_size=max_pool_size,
                            name='maxConv4')(conv4)

    conv5 = Conv1D(filters=num_filters,
                    kernel_size=filter_sizes[1],
                    padding='valid',
                    activation='relu',
                    name='conv5')(embedding_drop)

    maxConv5 = MaxPooling1D(pool_size=max_pool_size,
                            name='maxConv5')(conv5)

    h = concatenate([maxConv4, maxConv5])

    h_drop = Dropout(cnn_dropout)(h)

    # Add RNN layer
    rnn_output = GRU(rnn_hidden_size)(h_drop)

    # Add mlp layer
    mlp_output = Dense(units=mlp_hidden_size,
                        activation='relu',
                        kernel_initializer='he_normal',
                        kernel_constraint=maxnorm(max_norm),
                        bias_constraint=maxnorm(max_norm),
                        name='mlp')(rnn_output)

    mlp_output_drop = Dropout(mlp_dropout)(mlp_output)

    # Add output layer
    output = Dense(units=output_size,
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    # kernel_regularizer=regularizers.l2(self.l2_reg_lambda),
                    # kernel_constraint=maxnorm(self.max_norm),
                    # bias_constraint=maxnorm(self.max_norm),
                    name='output')(mlp_output_drop)

    # build the model
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss={'output':'binary_crossentropy'},
                optimizer=Adadelta(lr=base_lr, epsilon=1e-6, decay=decay_rate),
                metrics=["accuracy"])

    return model


def train(model, train_data, train_labels, test_data, test_labels, embeddings, train_dir):

    best_accu = 0.0
    # log writer
    train_loss_log = train_dir + "/train_loss.txt"  # training loss log file store directory
    val_loss_log = train_dir + "/val_loss.txt" # validation loss log file store directory
    test_loss_log = train_dir + "/test_loss.txt" # test loss and accuracy log file store directory
    train_log = open(train_loss_log, "w")
    val_log = open(val_loss_log, "w")
    test_log = open(test_loss_log, 'w')

    # print model architecture details
    print(model.summary())

    # start training
    for epoch in range(n_epochs):
        logging.info("Epoch %d out of %d", epoch + 1, n_epochs)
        his = model.fit(train_data, train_labels,
                    batch_size=batch_size,
                    validation_split=0.1,
                    shuffle=True,
                    epochs=1, verbose=1)
        logging.info('Epoch %d/%d\t%s' % (epoch + 1, n_epochs, str(his.history)))
        train_log.write('Epoch {}:'.format(epoch + 1) + str(his.history['loss']) + '\n')
        val_log.write('Epoch {}:'.format(epoch + 1) + str(his.history['val_loss']) + '\n')
        
        if his.history['val_acc'][0] > best_accu:
            best_accu = his.history['val_acc'][0]
            print("New best dev accuracy: {}! Saving model in {}".format(best_accu, train_dir))
            # serialize model to JSON
            model_json = model.to_json()
            with open(train_dir + "/model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(train_dir + "/model.h5")
            print("Saved model to disk")
            print("Starting to evaluate on test set!")
            test_loss, test_accu = model.evaluate(test_data, test_labels,
                                                batch_size=batch_size,
                                                verbose=1)
            logging.info("\nAverage Test Loss: {}, Average Test Accuracy: {}".format(test_loss, test_accu))
            test_log.write("\n{},{},{}".format(epoch + 1, test_loss, test_accu))

    return test_accu
        
def main():

    # read pre-trained embeddings
    embeddings = load_embeddings(embedding_path, 'word2vec')

    test_accus = [] # Collect test accuracy for each fold
    for i in xrange(n_folds):
        fold = i + 1
        logging.info('Fold {} of {}...'.format(fold, n_folds))
        # read data
        train_data, train_labels, test_data, test_labels, seq_len, vocab_size = load_data_MR(data_path, fold=fold)

        # update train directory according to fold number
        train_dir = base_train_dir + '/' + str(fold)
        # create train directory if not exist
        if not os.path.exists(train_dir):
            os.makedirs(train_dir)
        # create log file handler
        file_handler = logging.FileHandler(pjoin(train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # check whether the model has been trained, if not, create a new one
        if os.path.exists(train_dir + '/model.json'):
            # load json and create model
            json_file = open(train_dir + '/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights(train_dir + "/model.h5")
            model.compile(loss={'output':'binary_crossentropy'},
                        optimizer=Adadelta(lr=base_lr, epsilon=1e-6, decay=decay_rate),
                        metrics=["accuracy"])
            print("Loaded model from disk!")
        else:
            model = setup_model(embeddings, seq_len, vocab_size)
            print("Created a new model!")

        # train the model
        test_accu = train(model, train_data, train_labels, test_data, test_labels, embeddings, train_dir)

        # log test accuracy result
        logging.info("\nTest Accuracy for fold {}: {}".format(fold, test_accu))
        test_accus.append(test_accu)
    
    # write log of test accuracy for all folds
    test_accu_log = open(base_train_dir + "/final_test_accuracy.txt", 'w')
    test_accu_log.write('\n'.join(['Fold {} Test Accuracy: {}'.format(fold, test_accu) for fold, test_accu in enumerate(test_accus)]))
    test_accu_log.write('\nAvg test acc: {}'.format(np.mean(test_accus)))


if __name__ == "__main__":
    main()
