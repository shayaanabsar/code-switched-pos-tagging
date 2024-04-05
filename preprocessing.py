# tags NOUN, PRON, VERB, ADJ, ADV, DET, PART, X, S
from transformers import AutoTokenizer
from math import log
from collections import defaultdict
from dataclasses import dataclass
import csv
import torch
from math import e

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
start_token, end_token = tokenizer("").input_ids

def tanh(x, lam=4, mu=0.1):
	x *= mu
	return lam * (e**x - e**(-x)) / (e**x + e**(-x))

@dataclass
class PreProcessor:
	curr_seq  = [start_token]
	curr_tags = ['S']
	sequences = []
	cs_index  = []
	s_index   = []
	sequences = []
	num_sequences     = 0
	max_length        = 0
	alteration_points = 0
	seq_length        = 2
	num_tokens        = 2
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
		self.curr_seq.append(end_token)
		self.curr_tags.append('S')
		self.sequences.append([self.curr_seq, self.curr_tags])
		self.curr_seq  = [start_token]
		self.curr_tags = ['S']
		self.num_sequences += 1
		self.max_length = max(self.max_length, self.seq_length)
		self.alteration_points = 0
		self.tokens_in_each_language = defaultdict(lambda: 0) 
		self.seq_length = 2
		self.num_tokens = 2
		self.previous =  None


	def calculate_metrics(self):
		self.calculate_s_index()
		self.calculate_cs_index()
		self.reset_values()

	def read_data(self, file):
		with open(file) as f:
			reader = csv.reader(f, delimiter='\t')

			for row in reader:
				try:
					token, lang, tag = row
					self.update_values(lang)

					tokenized_token = tokenizer(token).input_ids[1:-1] # get rid of S and E tokens
					self.curr_seq.extend(tokenized_token)
					self.curr_tags.extend([tag] * len(tokenized_token))
					self.num_tokens += len(tokenized_token)

				except ValueError:
					self.calculate_metrics()
				

x = PreProcessor()
x.read_data('data.csv')
print(x.sequences[2])
