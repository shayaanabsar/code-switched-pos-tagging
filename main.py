from transformers import BertModel, AutoTokenizer
from preprocessing import *
from trainer import *
from torch import nn, save
import pandas as pd
import torch


pp = PreProcessor()
pp.read_data('dataset')
inputs, outputs, lang_tags = pp.create_lists()
	
b_s, b_e = pp.splitters['bengali.csv']
h_s, h_e = pp.splitters['hindi.csv'  ]
t_s, t_e = pp.splitters['telugu.csv' ]
b, h, t, o  =  (b_e-b_s), (h_e-h_s), (t_e-t_s), len(inputs)

b_csi, b_si = torch.tensor(pp.cs_index[b_s:b_e]), torch.tensor(pp.s_index[b_s:b_e])
h_csi, h_si = torch.tensor(pp.cs_index[h_s:h_e]), torch.tensor(pp.s_index[h_s:h_e])
t_csi, t_si = torch.tensor(pp.cs_index[t_s:t_e]), torch.tensor(pp.s_index[t_s:t_e])
o_csi, o_si = torch.tensor(pp.cs_index)         , torch.tensor(pp.s_index)

b_csi_m, b_si_m = torch.mean(b_csi).item(), torch.mean(b_si).item()
h_csi_m, h_si_m = torch.mean(h_csi).item(), torch.mean(h_si).item()
t_csi_m, t_si_m = torch.mean(t_csi).item(), torch.mean(t_si).item()
o_csi_m, o_si_m = torch.mean(o_csi).item(), torch.mean(o_si).item()

coef = torch.corrcoef(torch.stack([o_csi, o_si]))[0][1]
data = pd.DataFrame([
	['Hindi'  , h_csi_m, h_si_m, h], 
	['Bengali', b_csi_m, b_si_m, b], 
	['Telugu' , t_csi_m, t_si_m, t], 
	['Overall', o_csi_m, o_si_m, o]
])

data.columns = ['Language', 'Mean CM-Index', 'Mean S-Index', 'Count']
print(data, '\n\n')
print(f'PMCC between the CMI (Gambäck and Das) and the S-index: {coef:.5f}')


avoid_language = input('Enter language to avoid (blank if none): ')
if avoid_language != '':
	code = pp.lang_codes[avoid_language]

	allowed_indexes = []
	hidden_indexes  = []

	for i in range(pp.num_sequences):
		if lang_tags[i] != code:
			allowed_indexes.append(i)
		else:
			hidden_indexes.append(i)

	o = len(allowed_indexes)

	filtered_inputs, filtered_outputs, filtered_lang_tags = inputs[allowed_indexes], outputs[allowed_indexes], lang_tags[allowed_indexes]

	input_train, input_test, input_val    = filtered_inputs[:int(0.8*o)], filtered_inputs[int(0.8*o):int(0.9*o)], filtered_inputs[int(0.9*o):]
	output_train, output_test, output_val = filtered_outputs[:int(0.8*o)], filtered_outputs[int(0.8*o):int(0.9*o)], filtered_outputs[int(0.9*o):]
	hidden_input, hidden_output = inputs[hidden_indexes], outputs[hidden_indexes]

	output_tags = filtered_lang_tags[int(0.8*o):int(0.9*o)]
else:
	input_train, input_test, input_val    = inputs[:int(0.8*o)], inputs[int(0.8*o):int(0.9*o)], inputs[int(0.9*o):]
	output_train, output_test, output_val = outputs[:int(0.8*o)], outputs[int(0.8*o):int(0.9*o)], outputs[int(0.9*o):]

	output_tags = lang_tags[int(0.8*o):int(0.9*o)]

wandb.login()

xlm                     = BertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)
tokenizer               = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
xlm_output_size         = 768
num_tags                = len(pp.tagset)
batch_size              = 16
batch_accumulation      = 4
learning_rate           = 0.0001
epochs                  = 100
dropout_rate            = 0.4
sequence_length         = pp.max_length
device                  = 'cuda'

wandb.init(
	project="code-switched-pos-tagging",
	# track hyperparameters and run metadata
	config={
	"learning_rate": learning_rate,
	"architecture": "BERT",
	"batch_size": batch_size*batch_accumulation,
	"epochs": epochs,
	"hidden_language": avoid_language
	}
)


class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.xlm = xlm
		self.linear  = nn.Linear(xlm_output_size, num_tags)

	def forward(self, input, train=True):
		tokenized  = tokenizer.batch_encode_plus(input, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
		embeddings = self.xlm(**tokenized).last_hidden_state
		new_embeddings = []

		for i in range(len(input)): # The idea for this comes from https://github.com/Illinois-Linguistic-Data-Management/spanglish-pos-tagger/blob/main/models_archive/MBERT_morph_tagger.py
			buff = []

			sent_embedding     = embeddings[i][1:]
			new_sent_embedding = []
			word_ids = tokenized.word_ids(batch_index=i)[1:]

			for j, val in enumerate(word_ids):
				if val is None:
					break
				elif j+1 < len(word_ids) and val == word_ids[j+1]:
					buff.append(sent_embedding[j])
				elif len(buff) > 0:
					avg = torch.mean(torch.stack(buff), dim=0)
					buff = []
					new_sent_embedding.append(avg)
				else:
					new_sent_embedding.append(sent_embedding[j])

			new_sent_embedding = torch.stack(new_sent_embedding)
			new_embeddings.append(new_sent_embedding)

		# https://discuss.pytorch.org/t/stacking-a-list-of-tensors-whose-dimensions-are-unequal/31888/3
		max_rows = max(tensor.size(0) for tensor in new_embeddings)
		padded_data = [torch.nn.functional.pad(tensor, (0, 0, 0, max_rows - tensor.size(0))) for tensor in new_embeddings]
		new_embeddings = torch.stack(padded_data, dim=0)
		
		x = self.linear(new_embeddings)

		return x
		
