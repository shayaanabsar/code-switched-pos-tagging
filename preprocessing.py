from collections import defaultdict
from dataclasses import dataclass
from os import listdir, path
from math import log, e
import torch
import csv
import random

random.seed(35)

MAX_LENGTH  = 512
SPECIAL_TAG = 'S'

def tanh(x, lam=1, mu=0.5):
	x *= mu
	return lam * (e**x - e**(-x)) / (e**x + e**(-x))

@dataclass
class PreProcessor:
	sentence   = []
	curr_seq   = []
	curr_tags  = []
	lang_tags  = []
	sequences  = []
	cs_index   = []
	s_index    = []
	sequences  = []
	tagset     = {SPECIAL_TAG}
	splitters  = {}
	lang_codes = {}
	lang_count        = 1
	num_sequences     = 0
	max_length        = 0
	alteration_points = 0
	seq_length        = 0 # used only for calculating metrics
	num_tokens        = 2 # used in creating the shape of the tensor
	tokens_in_each_language = defaultdict(lambda: 0) 
	tag_counts              = defaultdict(lambda: 0)
	previous                =  None

	def update_values(self, l):
		self.seq_length += 1
		if self.previous is not None and self.previous != l: 
			self.alteration_points += 1
		self.tokens_in_each_language[l] += 1
		self.previous = l


	def calculate_cs_index(self):
		num_in_matrix = max(list(self.tokens_in_each_language.values())) 
		num_tokens_not_in_matrix = self.seq_length - num_in_matrix
		cs_index = (0.5 * num_tokens_not_in_matrix + 0.5 * self.alteration_points) / self.seq_length

		self.cs_index.append(cs_index)
	
	def calculate_s_index(self):
		num_in_matrix = max(list(self.tokens_in_each_language.values())) 
		l = log(self.seq_length / num_in_matrix)
		s_index = tanh(self.alteration_points) * l
		
		self.s_index.append(s_index)

	def reset_values(self):
		if self.num_tokens <= MAX_LENGTH:
			self.num_tokens += 1
			self.num_sequences += 1
			self.sequences.append([self.sentence, self.curr_tags])
			self.max_length = max(self.max_length, self.num_tokens)

		self.curr_seq  = []
		self.curr_tags = []
		self.alteration_points = 0
		self.tokens_in_each_language = defaultdict(lambda: 0) 
		self.seq_length = 2
		self.num_tokens = 2
		self.previous =  None
		self.sentence = []

	def calculate_metrics(self):
		if self.num_tokens <= MAX_LENGTH:
			self.calculate_s_index()
			self.calculate_cs_index()
			self.lang_tags.append(self.lang_count)
		self.reset_values()

	def read_data(self, folder):
		for file in listdir(folder):
			self.lang_codes[file] = self.lang_count
			with open(path.join(folder, file)) as f:
				reader = csv.reader(f, delimiter='\t')
				start = self.num_sequences
				for row in reader:
					try:
						token, lang, tag = row
						self.update_values(lang)
						#tokenized_token = tokenizer(token).input_ids[1:-1] # get rid of S and E tokens
						#self.curr_seq.extend(tokenized_token)
						#self.tag_counts[tag] += 1
						#tags = [tag] + ['S'] * (len(tokenized_token) - 1)
						self.curr_tags.append(tag)
						#self.num_tokens += len(tokenized_token)
						self.tagset.add(tag)
						self.sentence.append(token)
					except ValueError:
						#self.sentence = tokenizer.encode_plus(self.sentence, return_tensors="pt", is_split_into_words=True)
						self.calculate_metrics()
			self.splitters[file] = (start, self.num_sequences)
			self.lang_count += 1

	def create_lists(self):
		self.tagset = {'G_V': 0, 'G_SYM': 1, 'CC': 2, 'G_PRP': 3, 'DT': 4, 'G_PRT': 5, '@': 6, 'E': 7, '$': 8, '~': 9, '#': 10, 'PSP': 11, 'G_X': 12, 'G_R': 13, 'G_J': 14, 'G_N': 15, 'U': 16, 'S': 17, 'null': 18}
		#self.tagset = {'PAD': 0, 'ADJ': 1, 'ADP': 2, 'ADV': 3, 'CONJ': 4, 'DET': 5, 'NOUN': 6, 'NUM': 7, 'PART': 8, 'PART_NEG': 9, 'PRON': 10, 'PRON_WH': 11, 'PROPN': 12, 'VERB': 13, 'X': 14}
		
		num_tags = len(self.tagset)
		order = [i for i in range(self.num_sequences)]
		random.shuffle(order)

		inputs        = []
		outputs       = []
		shuffled_tags = torch.zeros((self.num_sequences,), dtype=int)

		# dodgy shuffle

		for i, val in enumerate(order):
			seq = self.sequences[val]

			input_sequence   = seq[0]
			tag_sequence     = seq[1]
			shuffled_tags[i] = self.lang_tags[val]

			inputs.append(input_sequence)

			encoded_tags = []

			for _, v in enumerate(tag_sequence):
				encoded_tags.append(self.tagset[v])

			outputs.append(encoded_tags)

		return inputs, outputs, shuffled_tags
