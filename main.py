from transformers import AutoTokenizer, XLMRobertaModel
from preprocessing import *
from trainer import *
from torch import nn, save
import pandas as pd
import torch


pp = PreProcessor()
pp.read_data('dataset')
input_tensor, output_tensor, lang_tags = pp.create_tensors()
	
b_s, b_e = pp.splitters['bengali.csv']
h_s, h_e = pp.splitters['hindi.csv'  ]
t_s, t_e = pp.splitters['telugu.csv' ]
b, h, t, o  =  (b_e-b_s), (h_e-h_s), (t_e-t_s), input_tensor.shape[0]

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

	filtered_input_tensor, filtered_output_tensor, filtered_lang_tags = input_tensor[allowed_indexes], output_tensor[allowed_indexes], lang_tags[allowed_indexes]

	input_train, input_test, input_val    = filtered_input_tensor[:int(0.8*o)], filtered_input_tensor[int(0.8*o):int(0.9*o)], filtered_input_tensor[int(0.9*o):]
	output_train, output_test, output_val = filtered_output_tensor[:int(0.8*o)], filtered_output_tensor[int(0.8*o):int(0.9*o)], filtered_output_tensor[int(0.9*o):]
	hidden_input, hidden_output = input_tensor[hidden_indexes], output_tensor[hidden_indexes]

	output_tags = filtered_lang_tags[int(0.8*o):int(0.9*o)]
else:
	input_train, input_test, input_val    = input_tensor[:int(0.8*o)], input_tensor[int(0.8*o):int(0.9*o)], input_tensor[int(0.9*o):]
	output_train, output_test, output_val = output_tensor[:int(0.8*o)], output_tensor[int(0.8*o):int(0.9*o)], output_tensor[int(0.9*o):]

	output_tags = lang_tags[int(0.8*o):int(0.9*o)]

wandb.login()

xlm_roberta             = XLMRobertaModel.from_pretrained("FacebookAI/xlm-roberta-base")
xlm_roberta_output_size = 768
cross_entropy_loss      = nn.CrossEntropyLoss()
num_tags                = output_train.shape[2]
batch_size              = 8
batch_accumulation      = 2
learning_rate           = 1e-8
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
		self.xlm_roberta = xlm_roberta
		self.linear  = nn.Linear(xlm_roberta_output_size, num_tags)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, input, train=True):
		roberta_logits = self.xlm_roberta(input).last_hidden_state
		x = self.linear(x)
		probabilities = self.softmax(x)

		return probabilities
		
