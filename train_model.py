from main import *

model = nn.DataParallel(Model()).to(device)
t = Trainer(model, learning_rate, device)
if avoid_language == '':
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val)
else:
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val, hidden_inputs, hidden_outputs)
save(model.state_dict(), 'model.pt')
artifact = wandb.Artifact(name='model', type='model')
artifact.add_file(local_path='model.pt')
r.log_artifact(artifact)
