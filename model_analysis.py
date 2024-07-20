import wandb
from main import *
print(pp.tagset)
counts = {l:[0, 0] for l in pp.lang_codes}
codes_to_langs = {pp.lang_codes[k]:k for k in pp.lang_codes}

run = wandb.init()

if avoid_language == '':
	artifact = run.use_artifact('shayaan-absar/code-switched-pos-tagging/model:v10', type='model')
elif avoid_language == 'bengali.csv':
	artifact = run.use_artifact('shayaan-absar/code-switched-pos-tagging/model:v3', type='model')
elif avoid_language == 'telugu.csv':
	artifact = run.use_artifact('shayaan-absar/code-switched-pos-tagging/model:v4', type='model')
elif avoid_language == 'hindi.csv':
	artifact = run.use_artifact('shayaan-absar/code-switched-pos-tagging/model:v6', type='model')
else:
	print('Invalid.')
	exit()

artifact_dir = artifact.download()

state_dict = torch.load(f'{artifact_dir}/model.pt', map_location='cpu')

new_state_dict = {}

for k, v in state_dict.items():
	if k.startswith('module.'):
		new_state_dict[k[7:]] = v
	else:
		new_state_dict[k] = v
		
model = Model()
model.load_state_dict(new_state_dict)

batch_size = 8
tokens = input_tensor.shape[-1]

def test(inputs, outputs):
	global batch_size, tokens, pp, counts

	right, total, n = 0, 0, 0


	for i in range(5):
		preds = torch.argmax(model(inputs[batch_size*n:batch_size*(n+1)]), dim=-1)
		acts  = torch.argmax(outputs[batch_size*n:batch_size*(n+1)], dim=-1)

		for sample_number in range(batch_size):
			for token_number in range(tokens):
				if acts[sample_number][token_number] != pp.tagset[SPECIAL_TAG]:
					if preds[sample_number][token_number] == acts[sample_number][token_number]:
						right += 1
						counts[codes_to_langs[output_tags[n*batch_size + sample_number].item()]][0] += 1
					counts[codes_to_langs[output_tags[n*batch_size + sample_number].item()]][1] += 1
					total += 1
		n += 1

		print(f'{(right/total)*100:.4f}')
		print(counts)

test(input_test, output_test)
if avoid_language != '':
	test(hidden_input, hidden_output)
