import numpy as np
import codecs 
import time
import string 
import torch
import os

'''
Batcher objects to represent getting batches from data 
'''
class Batcher(object):
	def __init__(self, config, input_type, labeled_file=None):
		'''
		param config: configuration object 
		param input_type: whether the batcher is for train/dev/test
		param labeled_file: labeled file to use for labels (default is specified in config )
		'''
		self.config = config

		if input_type == 'train':
			self.train_batch_size = config.train_batch_size
		else:
			self.dev_test_batch_size = config.dev_test_batch_size

		self.all_qry_id = []
		self.all_pos_id = []
		self.all_neg_id = []
		self.all_cnd_id = []
		self.all_lbl = []

		self.all_qry = []
		self.all_cnd = []

		if self.config.fold is None:
			if input_type == 'train':
				self.input_file = config.train_file
				self.train_load_data()
				self.train_shuffle_data()
			else:
				if input_type == 'dev':
					if labeled_file is not None:
						self.input_file = labeled_file
					else:
						self.input_file = config.dev_file
				else:
					# For partitioning test file to parallelize, so test_file won't be config but will be passed in
					if labeled_file is not None:
						self.input_file = labeled_file
					else:
						self.input_file = config.test_file
				self.dev_test_load_data()
		else:
			if input_type == 'train':
				self.input_file = os.path.join("data", "cross_validation", "fold_%d" % self.config.fold, "train.data")
				self.train_load_data()
				self.train_shuffle_data()
			else:
				if input_type == 'dev':
					if labeled_file is not None:
						self.input_file = labeled_file
					else:
						self.input_file = config.dev_file
						self.input_file = os.path.join("data", "cross_validation", "fold_%d" % self.config.fold, "dev.data")
				else:
					# For partitioning test file to parallelize, so test_file won't be config but will be passed in
					if labeled_file is not None:
						self.input_file = labeled_file
					else:
						self.input_file = config.test_file
						self.input_file = os.path.join("data", "cross_validation", "fold_%d" % self.config.fold, "test.data")
				self.dev_test_load_data()

		self.input_type = input_type
		self.start_idx = 0

	def get_next_dev_test_end_idx(self, cur_qry):
		'''
		Get the next batch end index such that every batch has the same query to calculate MAP without writing predictions
		param cur_qry: current query of the current batch 
		'''
		end_idx = self.start_idx

		# After breaking out of for loop, end_idx will point to first qry_id that doesn't match cur_qry_id
		while(end_idx < self.num_examples and self.all_qry_id[end_idx] == cur_qry):
			end_idx += 1

		return end_idx

	def get_dev_test_batches(self):
		'''
		Returns all the dev or test batches
		'''
		self.start_idx = 0
		cur_qry = self.all_qry_id[self.start_idx]

		while True:
			if self.start_idx >= self.num_examples:
				return
			else:
				end_idx = self.get_next_dev_test_end_idx(cur_qry)

				if end_idx > self.start_idx + self.dev_test_batch_size:
					end_idx = self.start_idx + self.dev_test_batch_size

				if end_idx < self.num_examples:
					cur_qry = self.all_qry[end_idx]

				end_block = (end_idx >= self.num_examples or self.all_qry_id[end_idx-1] != self.all_qry_id[end_idx])

				yield zip(self.all_qry[self.start_idx:end_idx], \
						  self.all_cnd[self.start_idx:end_idx]), \
					zip(self.all_qry_id[self.start_idx:end_idx], \
					  self.all_cnd_id[self.start_idx:end_idx]), \
					  self.all_lbl[self.start_idx:end_idx], \
					  end_block
				self.start_idx = end_idx

	def get_train_batches(self):
		'''
		Returns all the train batches, where each batch includes examples with the same query 
		'''
		while True:
			if self.start_idx > self.num_examples - self.train_batch_size: 
				self.start_idx = 0
				self.train_shuffle_data()
			else:
				end_idx = self.start_idx + self.train_batch_size

				yield zip(self.all_qry_id[self.start_idx:end_idx], \
					  self.all_pos_id[self.start_idx:end_idx], \
					  self.all_neg_id[self.start_idx:end_idx])
				self.start_idx = end_idx

	def train_shuffle_data(self):
		'''
		Shuffles the training data, maintining the permutation across query, positive, and negative ids
		'''
		perm = np.random.permutation(self.num_examples)  # perm of index in range(0, num_questions)
		assert len(perm) == self.num_examples
		self.all_qry_id, self.all_pos_id, self.all_neg_id = self.all_qry_id[perm], self.all_pos_id[perm], self.all_neg_id[perm]

	def dev_test_load_data(self):
		'''
		Loads and stores the ids for dev/test data
		'''
		with open(self.input_file, "r") as f:
			counter = 1
			for line in f.readlines():
				split = line.strip('\n').split("\t")

				if len(split) > 2:
					self.all_qry_id.append(split[0])
					self.all_cnd_id.append(split[1])
					self.all_lbl.append(int(split[2]))
					self.all_qry.append(split[0])
					self.all_cnd.append(split[1])
				else:
					print("Dev Line", line)

				counter += 1


		self.all_qry_id = np.asarray(self.all_qry_id)
		self.all_cnd_id = np.asarray(self.all_cnd_id)
		self.all_lbl = np.asarray(self.all_lbl, dtype=np.int32)
		self.all_qry = np.asarray(self.all_qry)
		self.all_cnd = np.asarray(self.all_cnd)
		self.num_examples = len(self.all_qry_id)

	def train_load_data(self):
		'''
		Loads and stores the ids for test data
		'''
		with open(self.input_file, "r") as inp:
			counter = 1

			for line in inp:
				split = line.split("\t")

				if len(split) > 2:
					self.all_qry_id.append(split[0])
					self.all_pos_id.append(split[1])
					self.all_neg_id.append(split[2])
				else:
					print("Train Line", line, counter)
				counter += 1

		assert(len(self.all_qry_id) == len(self.all_pos_id) and \
			   len(self.all_qry_id) == len(self.all_neg_id))

		self.all_qry_id = np.asarray(self.all_qry_id)
		self.all_pos_id = np.asarray(self.all_pos_id)
		self.all_neg_id = np.asarray(self.all_neg_id)
		self.num_examples = len(self.all_qry_id)
