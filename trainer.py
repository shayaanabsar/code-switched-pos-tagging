from dataclasses import dataclass, field
import plotly.express as px
import pandas as pd
import torch

@dataclass
class Trainer:
	model         : object 
	loss_function : object
	lr            : int
	train_metrics = []
	val_metrics   = []      

	def pass_batch(self, batch_size, inputs, outputs):
		max  = inputs.shape[0]
		idxs = torch.randint(max, (batch_size,))

		batch_inputs  = torch.stack([inputs[i]  for i in idxs])
		batch_outputs = torch.stack([outputs[i] for i in idxs])

		model_probabilities = self.model(batch_inputs).float()
		loss = self.loss_function(model_probabilities, batch_outputs)
		return loss
	
	def train(self, epochs, batch_size, t_inputs, t_outputs, v_inputs, v_outputs):
		optimizer = torch.optim.AdamW(self.model.parameters(), self.lr)

		for i in range(epochs):
			loss     = self.pass_batch(batch_size, t_inputs, t_outputs)
			val_loss = self.pass_batch(8, v_inputs, v_outputs)

			self.train_metrics.append(loss.item())
			self.val_metrics.append(val_loss.item())

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			print(f'Epoch number {i}. Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}')

	def plot_metrics(self):
		df = pd.DataFrame(dict(
			x = [i for i in range(len(self.train_metrics))],
			y = self.train_metrics
		)
		)

		fig = px.line(df, x="x", y="y", title='Training Loss')
		fig.show()