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

	run = wandb.init(
			project="code-switched-pos-tagging",

			# track hyperparameters and run metadata
			config={
			"learning_rate": self.lr,
			"architecture": "BERT",
			"batch_size": batch_size,
			"epochs": epochs,
			}
	)

	wandb.login()

	def pass_batch(self, batch_size, inputs, outputs):
		torch.cuda.empty_cache()
		max  = inputs.shape[0]
		idxs = torch.randint(max, (batch_size,))

		batch_inputs  = torch.stack([inputs[i]  for i in idxs]).to(self.device)
		batch_outputs = torch.stack([outputs[i] for i in idxs]).to(self.device)

		model_probabilities = self.model(batch_inputs).float()
		torch.cuda.empty_cache()
		loss = self.loss_function(model_probabilities, batch_outputs)
		torch.cuda.empty_cache()
		return loss
	
	def train(self, epochs, batch_size, batch_acc, t_inputs, t_outputs, v_inputs, v_outputs):
		optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)

		for i in range(epochs):
			loss = 0

			if i == epochs // 2:
				optimizer.param_groups[0]['lr'] /= 10
			for j in range(batch_acc):
				loss  += self.pass_batch(batch_size, t_inputs, t_outputs)
			loss /= batch_acc
			val_loss = self.pass_batch(batch_size // 2, v_inputs, v_outputs)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			self.train_metrics.append(loss.item())
			self.val_metrics.append(val_loss.item())

			wandb.log({
				'val-loss': val_loss,
				'loss'    : loss
			})
