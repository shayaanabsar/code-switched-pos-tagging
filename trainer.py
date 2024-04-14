from dataclasses import dataclass
import plotly.express as px
import pandas as pd
import wandb
import torch

@dataclass
class Trainer:
	model         : object 
	loss_function : object
	lr            : int
	device        : str
	train_metrics = []
	val_metrics   = []      

	wandb.login()

	def pass_batch(self, batch_size, inputs, outputs):
		max  = inputs.shape[0]
		idxs = torch.randint(max, (batch_size,))

		batch_inputs  = torch.stack([inputs[i]  for i in idxs]).to(self.device)
		batch_outputs = torch.stack([outputs[i] for i in idxs]).to(self.device)

		model_probabilities = self.model(batch_inputs).float()
		loss = self.loss_function(model_probabilities, batch_outputs)
		return loss
	
	def train(self, epochs, batch_size, t_inputs, t_outputs, v_inputs, v_outputs):
		optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)

		wandb.init(
			project="code-switched-pos-tagging",

			# track hyperparameters and run metadata
			config={
			"learning_rate": self.lr,
			"architecture": "BERT",
			"batch_size": batch_size,
			"epochs": epochs,
			}
		)

		for i in range(epochs):
			loss     = self.pass_batch(batch_size, t_inputs, t_outputs)
			val_loss = self.pass_batch(8, v_inputs, v_outputs)

			self.train_metrics.append(loss.item())
			self.val_metrics.append(val_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			wandb.log({
				'val-loss': val_loss,
				'loss'    : loss
			})
