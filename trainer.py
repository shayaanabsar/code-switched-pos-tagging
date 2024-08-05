from dataclasses import dataclass
import pandas as pd
import wandb
import torch

def masked_loss(output, target):
	alpha = torch.argmax(target, dim=-1) # Find the correct pos tag
	indices = torch.nonzero(alpha != 17) # Indices of all those that aren't a S
	
	i_indices, j_indices = indices[:, 0], indices[:, 1] 
	k_indices = alpha[i_indices, j_indices]

	r = output[i_indices, j_indices, k_indices] # the predicted probs for the correct tag
	l = ((1-r)**5) * torch.log(r+1e-8)
	l = torch.mean(l)

	return -l

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
		torch.cuda.empty_cache()
		max  = inputs.shape[0]
		idxs = torch.randint(max, (batch_size,))

		batch_inputs  = torch.stack([inputs[i]  for i in idxs]).to(self.device)
		batch_outputs = torch.stack([outputs[i] for i in idxs]).to(self.device)

		model_probabilities = self.model(batch_inputs).float()
		torch.cuda.empty_cache()
		loss = masked_loss(model_probabilities, batch_outputs)
		torch.cuda.empty_cache()
		return loss
	
	def train(self, epochs, batch_size, batch_acc, t_inputs, t_outputs, v_inputs, v_outputs, h_inputs=None, h_outputs=None):
		optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)
		scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

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

		for i in range(epochs):
			loss = 0
			for j in range(batch_acc):
				loss  += self.pass_batch(batch_size, t_inputs, t_outputs)
			loss /= batch_acc
			val_loss = self.pass_batch(batch_size, v_inputs, v_outputs)
			if h_inputs is not None:
				h_loss = self.pass_batch(batch_size, h_inputs, h_outputs)
			else:
				h_loss = 0
			scheduler.step(val_loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			self.train_metrics.append(loss.item())
			self.val_metrics.append(val_loss.item())

			wandb.log({
				'val-loss': val_loss,
				'loss'    : loss,
				'h-loss'  : h_loss,
				'lr': optimizer.param_groups[0]['lr']
			})
		return run