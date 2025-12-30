## Handwritten Digit Classification 

### Problem Description
Handwriting persists, even in the digital age. The ability for machines to correctly identify handwritten digits 
is valuable for tasks that require immediate action, such as routing letters or understanding the amount written 
on a check. It can also have historical significance when digitizing handwritten records from the past. 

### Context
This is the capstone project for DataTalks.Club's [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp). 
This capstone represents my first independent project with a convolutional neural network, which is building upon the lessons learned
in the zoomcamp.

### The Dataset
*Note: This dataset is downloaded in a special data format and has handling procedures that were not covered in the Zoomcamp.*

The dataset is the MNIST database of handwritten digits, which is a well-known dataset that is commonly used in beginner
machine learning projects. The dataset consists of 60,000 images in the training dataset and 10,000 images in the validation dataset.
The data can be downloaded via PyTorch using the following:

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



