import json
import os
import time
import cPickle
import numpy as np
import sys
import re
import operator
import fnmatch
from gensim.models.keyedvectors import KeyedVectors
from tensorflow.contrib import learn
from keras.preprocessing import sequence

# construc embedding vectors based on the google word2vec and vocabulary
def process_word2vec(word2vec_dir, vocab, save_path, random_init=True):

	# read pre-trained word embedddings from the binary file
	print('Loading google word2vec...')
	word2vec_path = word2vec_dir + '/GoogleNews-vectors-negative300.bin.gz'
	word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
	print('Word2vec loaded!')

	if random_init:
		word2vec = np.random.uniform(-0.25, 0.25, (len(vocab), 300))
	else:
		word2vec = np.zeros((len(vocab), 300))
	found = 0
	for idx, token in enumerate(vocab):
		try:
			vec = word_vectors[token]
		except:
			pass
		else:
			word2vec[idx, :] = vec
			found += 1

	del word_vectors

	print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab), word2vec_path))
	np.savez_compressed(save_path, word2vec=word2vec)
	print("saved trimmed word2vec matrix at: {}".format(save_path))


# construct embedding vectors according to the GloVe word vectors and vocabulary
def process_glove(glove_dir, glove_dim, vocab_dir, save_path, random_init=True):
	"""
	:param vocab_list: [vocab]
	:return:
	"""
	save_path = save_path + '.{}'.format(glove_dim)
	if not os.path.isfile(save_path + ".npz"):
		# read vocabulary
		with open(vocab_dir + '/vocabulary.pickle', 'rb') as f:
			vocab_map = cPickle.load(f)
			f.close()
		vocab_list = list(zip(*vocab_map)[0])

		glove_path = os.path.join(glove_dir, "glove.6B.{}d.txt".format(glove_dim))
		if random_init:
			glove = np.random.uniform(-0.25, 0.25, (len(vocab_list), glove_dim))
		else:
			glove = np.zeros((len(vocab_list), glove_dim))
		found = 0
		with open(glove_path, 'r') as fh:
			for line in fh.readlines():
				array = line.lstrip().rstrip().split(" ")
				word = array[0]
				vector = list(map(float, array[1:]))
				if word in vocab_list:
					idx = vocab_list.index(word)
					glove[idx, :] = vector
					found += 1
				if word.capitalize() in vocab_list:
					idx = vocab_list.index(word.capitalize())
					glove[idx, :] = vector
					found += 1
				if word.upper() in vocab_list:
					idx = vocab_list.index(word.upper())
					glove[idx, :] = vector
					found += 1

		print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
		np.savez_compressed(save_path, glove=glove)
		print("saved trimmed glove matrix at: {}".format(save_path))

def load_embeddings(dir, embedding_type):
	return np.load(dir)[embedding_type]

def clean_str(string):
	"""
	Tokenization/string cleaning for all datasets except for SST.
	Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
	"""
	string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
	string = re.sub(r"\'s", " \'s", string)
	string = re.sub(r"\'ve", " \'ve", string)
	string = re.sub(r"n\'t", " n\'t", string)
	string = re.sub(r"\'re", " \'re", string)
	string = re.sub(r"\'d", " \'d", string)
	string = re.sub(r"\'ll", " \'ll", string)
	string = re.sub(r",", " , ", string)
	string = re.sub(r"!", " ! ", string)
	string = re.sub(r"\(", " \( ", string)
	string = re.sub(r"\)", " \) ", string)
	string = re.sub(r"\?", " \? ", string)
	string = re.sub(r"\s{2,}", " ", string)
	return string.strip().lower()

# preprocess the MR datasets
def preprocess_data_and_labels_MR(positive_data_file_path, negative_data_file_path, save_path, pad_width=0):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	positive_examples = list(open(positive_data_file_path, "r").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file_path, "r").readlines())
	negative_examples = [s.strip() for s in negative_examples]

	# Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]

	# Generate labels
	positive_labels = [[1] for _ in positive_examples]
	negative_labels = [[0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)

	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = np.array(list(vocab_processor.fit_transform(x_text)))

	# pad the left and right with zeros
	if pad_width > 0:
		x_padded = np.lib.pad(x, ((0, 0), (pad_width, pad_width)), 'constant', constant_values=(0, 0))

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
	x_shuffled = x_padded[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# merge data and labels
	data_and_labels = zip(x_shuffled, y_shuffled)

	# save train data and labels
	with open(save_path + '/data_and_labels.pickle', 'w') as f:
		cPickle.dump(data_and_labels, f)
		f.close()

	# get vocabulary and save it
	# Extract word:id mapping from the object.
	vocab_dict = vocab_processor.vocabulary_._mapping
	# Sort the vocabulary dictionary on the basis of values(id)
	sorted_vocab_dict = sorted(vocab_dict.items(), key=operator.itemgetter(1))
	sorted_vocab = list(zip(*sorted_vocab_dict))[0]
	with open(save_path + '/vocabulary.pickle', 'w') as f:
		cPickle.dump(sorted_vocab, f)
		f.close()

	# Process word vector embeddings
	process_word2vec('../data', sorted_vocab, '../data/word2vec.trimmed')

# Extract a set of n-grams from a list of integers.
def create_ngram_set(input_list, ngram_value=2):
	
	return set(zip(*[input_list[i:] for i in range(ngram_value)]))


# Augment the input list of list (sequences) by appending n-grams values.
def add_ngram(sequences, token_indice, ngram_range=2):

	new_sequences = []
	for input_list in sequences:
		new_list = input_list[:]
		for ngram_value in range(2, ngram_range + 1):
			for i in range(len(new_list) - ngram_value + 1):
				ngram = tuple(new_list[i:i + ngram_value])
				if ngram in token_indice:
					new_list.append(token_indice[ngram])
		new_sequences.append(new_list)

	return new_sequences

# preprocess the MR datasets especially for fasttext model
def preprocess_data_and_labels_MR_fasttext(positive_data_file_path, negative_data_file_path, save_path, ngram_range=1, pad_width=0):
	"""
	Loads MR polarity data from files, splits the data into words and generates labels.
	Returns split sentences and labels.
	"""
	# Load data from files
	positive_examples = list(open(positive_data_file_path, "r").readlines())
	positive_examples = [s.strip() for s in positive_examples]
	negative_examples = list(open(negative_data_file_path, "r").readlines())
	negative_examples = [s.strip() for s in negative_examples]

	# Split by words
	x_text = positive_examples + negative_examples
	x_text = [clean_str(sent) for sent in x_text]

	# Generate labels
	positive_labels = [[1] for _ in positive_examples]
	negative_labels = [[0] for _ in negative_examples]
	y = np.concatenate([positive_labels, negative_labels], 0)

	# Build vocabulary
	max_document_length = max([len(x.split(" ")) for x in x_text])
	vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
	x = list(vocab_processor.fit_transform(x_text))

	# Extract word:id mapping from the object.
	vocab_dict = vocab_processor.vocabulary_._mapping
	max_features = len(vocab_dict)

	# remove filled <UNK>, i.e., 0 index
	x = [filter(lambda a: a != 0, line) for line in x]
	print('Average sequence length before adding n-grams: {}'.format(np.mean(list(map(len, x)), dtype=int)))

	# Add n-grams...
	if ngram_range > 1:
		print('Adding {}-gram features'.format(ngram_range))
		# Create set of unique n-gram from the training set.
		ngram_set = set()
		for input_list in x:
			for i in range(2, ngram_range + 1):
				set_of_ngram = create_ngram_set(input_list, ngram_value=i)
				ngram_set.update(set_of_ngram)

		# Dictionary mapping n-gram token to a unique integer.
		# Integer values are greater than max_features in order
		# to avoid collision with existing features.
		start_index = max_features + 1
		token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
		indice_token = {token_indice[k]: k for k in token_indice}

		# Augmenting with n-grams features
		x = add_ngram(x, token_indice, ngram_range)
		print('Average sequence length after adding n-grams: {}'.format(np.mean(list(map(len, x)), dtype=int)))

		# pad sequence
		x = np.array(sequence.pad_sequences(x, padding='post'))
		print('x shape:', x.shape)

	# pad the left and right with zeros
	if pad_width > 0:
		x_padded = np.lib.pad(x, ((0, 0), (pad_width, pad_width)), 'constant', constant_values=(0, 0))

	# Randomly shuffle data
	np.random.seed(10)
	shuffle_indices = np.random.permutation(np.arange(x_padded.shape[0]))
	x_shuffled = x_padded[shuffle_indices]
	y_shuffled = y[shuffle_indices]

	# merge data and labels
	data_and_labels = zip(x_shuffled, y_shuffled)

	# save train data and labels
	with open(save_path, 'w') as f:
		cPickle.dump(data_and_labels, f)
		f.close()


def load_data_MR(file_dir, fold=1):
	print ("Loading datasets...")

	# read train data and labels
	with open(file_dir + '/data_and_labels.pickle', 'r') as f:
		data_and_labels = cPickle.load(f)
		f.close()

	# Split train/test set
	test_sample_index_s = int((fold - 1) / 10.0 * float(len(data_and_labels)))
	test_sample_index_e = int(fold / 10.0 * float(len(data_and_labels)))
	train_data_and_labels = data_and_labels[:test_sample_index_s] + data_and_labels[test_sample_index_e:]
	test_data_and_labels = data_and_labels[test_sample_index_s:test_sample_index_e]	

	# Split data and labels
	train_data, train_labels = zip(*train_data_and_labels)
	train_data, train_labels = np.array(train_data), np.array(train_labels)
	test_data, test_labels = zip(*test_data_and_labels)
	test_data, test_labels = np.array(test_data), np.array(test_labels)

	# read vocabulary
	with open(file_dir + '/vocabulary.pickle', 'r') as f:
		vocab = cPickle.load(f)
		f.close()

	seq_len = train_data.shape[1]	
	vocab_size = len(vocab)

	return (train_data, train_labels, test_data, test_labels, seq_len, vocab_size)

def load_data_MR_fasttext(file_path, fold=1):
	print ("Loading datasets...")

	# read train data and labels
	with open(file_path, 'r') as f:
		data_and_labels = cPickle.load(f)
		f.close()

	# Split train/test set
	test_sample_index_s = int((fold - 1) / 10.0 * float(len(data_and_labels)))
	test_sample_index_e = int(fold / 10.0 * float(len(data_and_labels)))
	train_data_and_labels = data_and_labels[:test_sample_index_s] + data_and_labels[test_sample_index_e:]
	test_data_and_labels = data_and_labels[test_sample_index_s:test_sample_index_e]	

	# Split data and labels
	train_data, train_labels = zip(*train_data_and_labels)
	train_data, train_labels = np.array(train_data), np.array(train_labels)
	test_data, test_labels = zip(*test_data_and_labels)
	test_data, test_labels = np.array(test_data), np.array(test_labels)

	seq_len = train_data.shape[1]	
	vocab_size = max([np.amax(train_data), np.amax(test_data)]) + 1

	return (train_data, train_labels, test_data, test_labels, seq_len, vocab_size)

# preprocess the AskaPatient dataset
def preprocess_data_and_labels_AAP(data_file_path, save_path):
	def merge_folds(data_file_path, save_path):
		# merge all the separated folds into one file
		train = []
		val = []
		test = []
		for file in os.listdir(data_file_path):
			if fnmatch.fnmatch(file, '*train.txt'):
				train += (open(data_file_path + '/' + file, 'r').readlines())
			elif fnmatch.fnmatch(file, '*validation.txt'):
				val += (open(data_file_path + '/' + file, 'r').readlines())
			else:
				test += (open(data_file_path + '/' + file, 'r').readlines())

		open(save_path + '/train.txt', 'w').write(''.join(train))
		open(save_path + '/val.txt', 'w').write(''.join(val))
		open(save_path + '/test.txt', 'w').write(''.join(test))
		print len(train+val+test)

	merge_folds(data_file_path, save_path)


def create_batches(data, labels, batch_size, shuffle=True):
	
	# Generates a batch iterator for a dataset.
	data_and_labels = np.array(zip(data, labels))
	data_size = len(data)
	num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
	# Shuffle the data
	if shuffle:
		np.random.seed(11)
		shuffle_indices = np.random.permutation(np.arange(data_size))
		shuffled_data = data_and_labels[shuffle_indices]
	else:
		shuffled_data = data_and_labels

	# create batches
	batches = []
	for batch_num in range(num_batches_per_epoch):
		start_index = batch_num * batch_size
		end_index = min((batch_num + 1) * batch_size, data_size)
		batches.append(shuffled_data[start_index:end_index])

	return batches

class Progbar(object):
	"""
	Progbar class copied from keras (https://github.com/fchollet/keras/)
	Displays a progress bar.
	# Arguments
		target: Total number of steps expected.
		interval: Minimum visual progress update interval (in seconds).
	"""

	def __init__(self, target, width=30, verbose=1):
		self.width = width
		self.target = target
		self.sum_values = {}
		self.unique_values = []
		self.start = time.time()
		self.total_width = 0
		self.seen_so_far = 0
		self.verbose = verbose

	def update(self, current, values=None, exact=None):
		"""
		Updates the progress bar.
		# Arguments
			current: Index of current step.
			values: List of tuples (name, value_for_last_step).
				The progress bar will display averages for these values.
			exact: List of tuples (name, value_for_last_step).
				The progress bar will display these values directly.
		"""
		values = values or []
		exact = exact or []

		for k, v in values:
			if k not in self.sum_values:
				self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
				self.unique_values.append(k)
			else:
				self.sum_values[k][0] += v * (current - self.seen_so_far)
				self.sum_values[k][1] += (current - self.seen_so_far)
		for k, v in exact:
			if k not in self.sum_values:
				self.unique_values.append(k)
			self.sum_values[k] = [v, 1]
		self.seen_so_far = current

		now = time.time()
		if self.verbose == 1:
			prev_total_width = self.total_width
			sys.stdout.write("\b" * prev_total_width)
			sys.stdout.write("\r")

			numdigits = int(np.floor(np.log10(self.target))) + 1
			barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
			bar = barstr % (current, self.target)
			prog = float(current)/self.target
			prog_width = int(self.width*prog)
			if prog_width > 0:
				bar += ('='*(prog_width-1))
				if current < self.target:
					bar += '>'
				else:
					bar += '='
			bar += ('.'*(self.width-prog_width))
			bar += ']'
			sys.stdout.write(bar)
			self.total_width = len(bar)

			if current:
				time_per_unit = (now - self.start) / current
			else:
				time_per_unit = 0
			eta = time_per_unit*(self.target - current)
			info = ''
			if current < self.target:
				info += ' - ETA: %ds' % eta
			else:
				info += ' - %ds' % (now - self.start)
			for k in self.unique_values:
				if isinstance(self.sum_values[k], list):
					info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
				else:
					info += ' - %s: %s' % (k, self.sum_values[k])

			self.total_width += len(info)
			if prev_total_width > self.total_width:
				info += ((prev_total_width-self.total_width) * " ")

			sys.stdout.write(info)
			sys.stdout.flush()

			if current >= self.target:
				sys.stdout.write("\n")

		if self.verbose == 2:
			if current >= self.target:
				info = '%ds' % (now - self.start)
				for k in self.unique_values:
					info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
				sys.stdout.write(info + "\n")

	def add(self, n, values=None):
		self.update(self.seen_so_far+n, values)

if __name__=="__main__":
	preprocess_data_and_labels_MR('../data/rt-polarity.pos', '../data/rt-polarity.neg', '../data', pad_width=4)
	# preprocess_data_and_labels_MR_fasttext('../data/rt-polarity.pos', '../data/rt-polarity.neg', '../data/fasttext_data_and_labels.pickle', 
	# 										ngram_range=3, pad_width=4)

