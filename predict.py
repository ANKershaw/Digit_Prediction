import numpy as np
import onnxruntime as ort
import torch.nn as nn
import torchvision.transforms as transforms
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from io import BytesIO
from PIL import Image
app = FastAPI(title="digit_predict")


class DigitClassifierScratch(nn.Module):
	def __init__(self, num_classes=10, dropout_rate=0.3, inner_size=240):
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


def image_transform(img):
	image_transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])
	img = Image.open(BytesIO(img))
	img = image_transforms(img)
	img = img.unsqueeze(0)
	# ONNX Runtime wants numpy float32
	img = img.numpy().astype(np.float32)
	return img


def predict_single(img):
	# prepare the image for prediction
	img = image_transform(img)
	
	onnx_path = 'digit_classifier_scratch.onnx'
	session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
	
	inputs = session.get_inputs()
	outputs = session.get_outputs()
	
	input_name = inputs[0].name
	output_name = outputs[0].name
	
	(logits,) = session.run([output_name], {
		input_name: img
	})
	
	result = int(np.argmax(logits, axis=1)[0])
	return result


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
	prediction = ''
	try:
		img = file.file.read()
		prediction = predict_single(img)
		
		
	except Exception as e:
		print(e)
		raise HTTPException(status_code=500, detail=f'Something went wrong: {e}')
	finally:
		file.file.close()
	
	return {
		"prediction": f'{prediction}'
	}

@app.get("/health")
def health():
	return {
		"status": "healthy"
	}


if __name__ == "__main__":
	uvicorn.run(app, host="0.0.0.0", port=9696)