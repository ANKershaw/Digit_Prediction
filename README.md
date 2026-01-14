## Handwritten Digit Classification 

### Problem Description
Writing by hand persists, even in the digital age. As more and more processing tasks are ported from humans to machines, the 
ability for a computer system to easily and correctly identify human writing becomes paramount. There are tasks of immediate 
importance, such as when postal services around the globe need to quickly and correctly route the millions of postal items received in a day, 
or when banks must process millions of checks and route the exactly correct amount of money. Failures in these two fields alone are 
inconvenient on a good day and potentially harmful on a bad one.

### Context
This is the capstone project for DataTalks.Club's [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).
This capstone represents my first independent project with a neural network, which is building upon the lessons learned over 
the past four months. 

### The Dataset
*Note: This dataset is downloaded in a special data format and has handling procedures that were not covered in the Zoomcamp.*

The dataset is the MNIST database of handwritten digits, which is a well-known dataset that is commonly used in beginner
machine learning projects and is sometimes considered the "hello world" example project. The dataset consists of 60,000 
images in the training dataset and 10,000 images in the validation dataset. The data can be downloaded via PyTorch using the following:

```python
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Define a transformation to convert images to PyTorch tensors
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)) # Standard normalization for MNIST
])

# Download and load the training data
trainset = MNIST(
    root='./data',      # Directory where data will be saved
    train=True,         # Request the training subset
    download=True,      # Download the data if it's not already present
    transform=transform
)

# Download and load the test data
testset = MNIST(
    root='./data',      # Directory where data will be saved
    train=False,        # Request the test subset
    download=True,      # Download the data if it's not already present
    transform=transform
)
```
The following will be downloaded:
* Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
* Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
* Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
* Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

Which contain:
* train-images-idx3-ubyte.gz: Training set images (60,000 images, 28x28 pixels, 3 dimensions).
* train-labels-idx1-ubyte.gz: Training set labels (60,000 labels, 1 dimension).
* t10k-images-idx3-ubyte.gz: Test set images.
* t10k-labels-idx1-ubyte.gz: Test set labels. 

The MSINT class, with the transform functions, take the images and create the tensors required for later processing. 
In the Machine Learning Zoomcamp, this replaces the Dataset class we built that ultimately loads the image datasets and applies the transformations.



### Project Workflow




### Requirements

In order to run this project you'll need to clone the repo and install the following (zoomcamp participants should already have these installed.)
kind: [https://kind.sigs.k8s.io/docs/user/quick-start/](https://kind.sigs.k8s.io/docs/user/quick-start/)
docker: [https://docs.docker.com/desktop/](https://docs.docker.com/desktop/)
uv: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)
python 3.13: [https://www.python.org/downloads/](https://www.python.org/downloads/)

### Dependency Management
Install packages: `uv sync --locked`

### Deploy


### Example post request
```shell
curl -X 'POST' \
  'http://0.0.0.0:9696/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@mnist_0_label_5.png;type=image/png'
```