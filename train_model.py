from main import *

model = nn.DataParallel(Model()).to(device)
t = Trainer(model, cross_entropy_loss, learning_rate, device, loss_weighting)
if avoid_language == '':
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val)
else:
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val, hidden_input, hidden_output)
save(model.state_dict(), 'model.pt')
artifact = wandb.Artifact(name='model', type='model')
artifact.add_file(local_path='model.pt')
r.log_artifact(artifact)
