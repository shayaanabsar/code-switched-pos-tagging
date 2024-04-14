from transformers import AutoTokenizer, AutoModelForMaskedLM
from preprocessing import *
from trainer import *
from torch import nn
import pandas as pd
import torch


pp = PreProcessor()
pp.read_data('dataset')
input_tensor, output_tensor = pp.create_tensors()

b_s, b_e = pp.splitters['bengali.csv']
h_s, h_e = pp.splitters['hindi.csv'  ]
t_s, t_e = pp.splitters['telugu.csv' ]

b_input, b_output = input_tensor[b_s:b_e], output_tensor[b_s:b_e]
h_input, h_output = input_tensor[h_s:h_e], output_tensor[h_s:h_e]
t_input, t_output = input_tensor[t_s:t_e], output_tensor[t_s:t_e]

b, h, t, o  =  (b_e-b_s), (h_e-h_s), (t_e-t_s), input_tensor.shape[0]

b_input_train, b_input_test, b_input_val    = b_input[:int(0.8*b)], b_input[int(0.8*b):int(0.9*b)], b_input[int(0.9*b):]
h_input_train, h_input_test, h_input_val    = b_input[:int(0.8*h)], b_input[int(0.8*h):int(0.9*h)], h_input[int(0.9*h):]
t_input_train, t_input_test, t_input_val    = t_input[:int(0.8*t)], t_input[int(0.8*b):int(0.9*t)], t_input[int(0.9*t):]

b_output_train, b_output_test, b_output_val = b_output[:int(0.8*b)], b_output[int(0.8*b):int(0.9*b)], b_output[int(0.9*b):]
h_output_train, h_output_test, h_output_val = b_output[:int(0.8*h)], b_output[int(0.8*h):int(0.9*h)], h_output[int(0.9*h):]
t_output_train, t_output_test, t_output_val = t_output[:int(0.8*t)], t_output[int(0.8*b):int(0.9*t)], t_output[int(0.9*t):]

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

wandb.login()

xlm_roberta             = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
xlm_roberta_output_size = 250002
cross_entropy_loss      = nn.CrossEntropyLoss()
num_tags                = b_output_train.shape[2]
batch_size              = 2
learning_rate           = 0.01
epochs                  = 100
dropout_rate            = 0.2
sequence_length         = pp.max_length
device                  = 'cuda'

wandb.init(
	project="code-switchwed-pos-tagging",
	# track hyperparameters and run metadata
	config={
	"learning_rate": learning_rate,
	"architecture": "BERT",
	"batch_size": batch_size,
	"epochs": epochs,
	}
)

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		self.xlm_roberta    = xlm_roberta
		self.dropout        = nn.Dropout(dropout_rate)
		self.linear         = nn.Linear(xlm_roberta_output_size, num_tags)
		self.batch_norm     = nn.BatchNorm1d(num_features=sequence_length)
		self.softmax        = nn.Softmax(dim=-1)

	def forward(self, input):
		roberta_logits      = self.xlm_roberta(input).logits
		dropout_logits      = self.dropout(roberta_logits)
		model_logits        = self.linear(dropout_logits)
		normalised_logits   = self.batch_norm(model_logits)
		model_probabilities = self.softmax(normalised_logits) 

		return model_probabilities
	
test = Trainer(Model().to(device), cross_entropy_loss, 0.01, device)
test.train(10, batch_size, b_input_test, b_output, b_input_val, b_output_val)
