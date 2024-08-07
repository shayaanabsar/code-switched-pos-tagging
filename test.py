from main import *
#from trainer import *

#model = Model()
#model.train()
#model.xlm.train()
#print(input_train[0])
test_in, test_out = [input_test[0], input_test[2]], [output_test[1], output_test[2]]

#optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)


#for i in range(50):
#	optimizer.zero_grad()

#	model_probabilities = model(test_in)
#	loss = masked_loss(model_probabilities, test_out)

#	loss.backward()
#	optimizer.step()
#	print(i, loss)


from main import *

model = Model()
t = Trainer(model, learning_rate, 'cpu')
if avoid_language == '':
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val)
else:
	r = t.train(epochs, batch_size, batch_accumulation, input_train, output_train, input_val, output_val, hidden_input, hidden_output)

predicted = torch.argmax(t.model(test_in), dim=-1)
target = test_out
right = 0
total = 0
for i in range(len(target)):
	for j in range(len(target[i])):
		if predicted[i][j] == target[i][j]:
			right += 1
print(right/total)