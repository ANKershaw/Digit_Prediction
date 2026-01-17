import numpy as np
import onnxruntime as ort
from keras_image_helper import create_preprocessor
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from io import BytesIO
from PIL import Image
app = FastAPI(title="digit_predict")

def preprocess_pytorch_style(img):
    img = Image.open(BytesIO(img)).convert("L")
    
    # PIL → NumPy (H, W), uint8 → float32
    img = np.array(img, dtype=np.float32)
    
    # Add batch + channel dims → (1, 1, H, W)
    img = np.expand_dims(img, axis=(0, 1))
    
    img = img / 255.0

    # Normalize
    img = (img - 0.1307) / 0.3081

    return img.astype(np.float32)


def image_transform(img):

    preprocessor = create_preprocessor(
        preprocess_pytorch_style,
        target_size=(28, 28)
    )
    img = preprocessor.preprocess(img)
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
async def predict(file: UploadFile = File(..., example="./test_images/mnist_0_label_5.png")) -> dict[str, str]:
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
    uvicorn.run(app, host="0.0.0.0", port=8080)