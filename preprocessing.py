from transformers import AutoTokenizer
from collections import defaultdict
from dataclasses import dataclass
from os import listdir, path
from math import log, e
import torch
import csv
import random

random.seed(35)

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
START_TOKEN, END_TOKEN = tokenizer("").input_ids
SPECIAL_TAG = 'S'
MAX_LENGTH  = 512

def tanh(x, lam=1, mu=0.5):
	x *= mu
	return lam * (e**x - e**(-x)) / (e**x + e**(-x))

@dataclass
class PreProcessor:
	curr_seq   = [START_TOKEN]
	curr_tags  = [SPECIAL_TAG]
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
			self.curr_seq.append(END_TOKEN)
			self.curr_tags.append(SPECIAL_TAG)
			self.num_tokens += 1
			self.num_sequences += 1
			self.sequences.append([self.curr_seq, self.curr_tags])
			self.max_length = max(self.max_length, self.num_tokens)

		self.curr_seq  = [START_TOKEN]
		self.curr_tags = [SPECIAL_TAG]
		self.alteration_points = 0
		self.tokens_in_each_language = defaultdict(lambda: 0) 
		self.seq_length = 2
		self.num_tokens = 2
		self.previous =  None


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
						tokenized_token = tokenizer(token).input_ids[1:-1] # get rid of S and E tokens
						self.curr_seq.extend(tokenized_token)
						tags = [tag] + ['S'] * (len(tokenized_token) - 1)
						self.curr_tags.extend(tags) # add the same tag for each 
						self.num_tokens += len(tokenized_token)
						self.tagset.add(tag)
			
					except ValueError:
						self.calculate_metrics()
			self.splitters[file] = (start, self.num_sequences)
			self.lang_count += 1

	def create_tensors(self):
		self.tagset = {'G_V': 0, 'G_SYM': 1, 'CC': 2, 'G_PRP': 3, 'DT': 4, 'G_PRT': 5, '@': 6, 'E': 7, '$': 8, '~': 9, '#': 10, 'PSP': 11, 'G_X': 12, 'G_R': 13, 'G_J': 14, 'G_N': 15, 'U': 16, 'S': 17, 'null': 18}
		num_tags = len(self.tagset)
		order = [i for i in range(self.num_sequences)]
		random.shuffle(order)

		inputs        = torch.zeros((self.num_sequences, self.max_length), dtype=int)
		outputs       = torch.zeros((self.num_sequences, self.max_length, num_tags), dtype=int)
		shuffled_tags = torch.zeros((self.num_sequences,), dtype=int)
		# dodgy shuffle
		for i, val in enumerate(order):
			seq = self.sequences[val]

			input_sequence = seq[0]
			tag_sequence   = seq[1]
			shuffled_tags[i]   = self.lang_tags[val]

			for j, v in enumerate(input_sequence):
				curr_tag = tag_sequence[j]

				inputs[i, j] = v
				outputs[i, j, self.tagset[curr_tag]] = 1
			

			outputs[i, j:self.max_length, self.tagset[SPECIAL_TAG]] = 1 # Pad all the remaining tags with the S tag.

		return inputs, outputs.float(), shuffled_tags
