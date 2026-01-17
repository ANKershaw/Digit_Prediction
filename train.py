import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

class DigitClassifierScratch(nn.Module):
	def __init__(self, num_classes=10, dropout_rate=0.0, inner_size=120):
		super(DigitClassifierScratch, self).__init__()
		
		# for extracting features
		self.layer1 = nn.Sequential(
			nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)
		
		self.layer3 = nn.Sequential(
			nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.AdaptiveAvgPool2d((1, 1))
		)
		
		# dense layers
		self.flatten = nn.Flatten(1)
		
		self.inner_layer = nn.Linear(
			in_features=64,
			out_features=inner_size
		)
		self.ReLU = nn.ReLU()
		self.dropout = nn.Dropout(dropout_rate)
		self.output_layer = nn.Linear(
			in_features=inner_size,
			out_features=num_classes
		)
	
	def forward(self, x):
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		
		x = self.flatten(x)
		x = self.inner_layer(x)
		x = self.ReLU(x)
		x = self.dropout(x)
		x = self.output_layer(x)
		
		return x


def main():
	# establishing the seed values
	SEED = 42
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	
	# offload tensor computations and neural network inference to the GPU (Apple’s Metal Performance Shaders (MPS))
	if torch.backends.mps.is_built() and torch.backends.mps.is_available():
		device = torch.device("mps")
		torch.mps.manual_seed(SEED)
	
	elif torch.cuda.is_available():
		device = torch.device("cuda")
	
	else:
		device = torch.device("cpu")
	
	
	# This transform is for the "from scratch" model and uses the standard MNIST normalize values
	transform_scratch = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	
	# Download and load the training data
	# The MNIST class provides a way to download and apply transformations to the MNIST images
	
	# For Model 2
	trainset_scratch = MNIST(
		root='./data_scratch',  # Directory where data will be saved
		train=True,  # Request the training subset
		download=True,  # Download the data if it's not already present
		transform=transform_scratch
	)
	
	testset_scratch = MNIST(
		root='./data_scratch',  # Directory where data will be saved
		train=False,  # Request the test subset
		download=True,  # Download the data if it's not already present
		transform=transform_scratch
	)
	
	train_loader_scratch = DataLoader(trainset_scratch, batch_size=32, shuffle=True)
	val_loader_scratch = DataLoader(testset_scratch, batch_size=32, shuffle=False)
	
	# set the criterion model to measure the error
	criterion = nn.CrossEntropyLoss()
	
	# creates new instances of the model
	def make_model_scratch(learning_rate, dropout_rate, inner_size):
		model = DigitClassifierScratch(num_classes=10, dropout_rate=dropout_rate, inner_size=inner_size)
		model.to(device)
		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		return model, optimizer
	
	highest_accuracy = 0
	
	model, optimizer = make_model_scratch(learning_rate=.001, dropout_rate=0.3, inner_size=240)
	
	for epoch in range(10):
		model.train()  # Set the model to training mode
		running_loss = 0.0
		correct = 0
		total = 0
		
		# Iterate over the training data
		for inputs, labels in train_loader_scratch:
			# Move data to the specified device (GPU or CPU)
			inputs, labels = inputs.to(device), labels.to(device)
			
			# Zero the parameter gradients to prevent accumulation
			optimizer.zero_grad()
			# Forward pass
			outputs = model(inputs)
			# Calculate the loss
			loss = criterion(outputs, labels)
			# Backward pass and optimize
			loss.backward()
			optimizer.step()
			
			# Accumulate training loss
			running_loss += loss.item()
			# Get predictions
			_, predicted = torch.max(outputs.data, 1)
			# Update total and correct predictions
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
		
		# Calculate average training loss and accuracy
		train_loss = running_loss / len(train_loader_scratch)
		train_acc = correct / total
		
		# Validation phase
		model.eval()  # Set the model to evaluation mode
		val_loss = 0.0
		val_correct = 0
		val_total = 0
		
		# Disable gradient calculation for validation
		with torch.no_grad():
			# Iterate over the validation data
			for inputs, labels in val_loader_scratch:
				# Move data to the specified device (GPU or CPU)
				inputs, labels = inputs.to(device), labels.to(device)
				# Forward pass
				outputs = model(inputs)
				# Calculate the loss
				loss = criterion(outputs, labels)
				
				# Accumulate validation loss
				val_loss += loss.item()
				# Get predictions
				_, predicted = torch.max(outputs.data, 1)
				# Update total and correct predictions
				val_total += labels.size(0)
				val_correct += (predicted == labels).sum().item()
		
		# Calculate average validation loss and accuracy
		val_loss /= len(val_loader_scratch)
		val_acc = val_correct / val_total
		
		# Print learning rate results
		print(f'_______________________________________')
		print(f'  Learning rate: .001                 ')
		print(f'  Dropout rate: 0.3      ')
		print(f'  Inner layer size: 240     ')
		print(f'_______________________________________')
		print(f'   → Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
		print(f'   → Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
		if highest_accuracy < val_acc:
	
			dummy_input = (torch.randn(1, 1, 28, 28, device="cpu"),)
			model_cpu = model.to("cpu")
			model_cpu.eval()
			with torch.no_grad():
				torch.onnx.export(model_cpu,
				                  dummy_input,
				                  "digit_classifier_scratch.onnx",
				                  external_data=False
				                  )
			print(f'  Saving model.     \n')
			highest_accuracy = val_acc
			model.to(device)
			
	print('All epochs complete.')
		
if __name__ == "__main__":
	main()
