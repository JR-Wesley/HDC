import torch
import torchmetrics
from rich.progress import track

import sys
sys.path.insert(0, '/home/maria/py/dl/my_d2l')
from my_package import preData, fig, hdc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_path = "../../data"
batch_size = 256
train_iter, test_iter = preData.load_data_mnist(path=dataset_path, batch_size=batch_size)

dim = 1024
datatype = 'binary' # 'binary' {0, 1} 'bipolar' {-1, 1}
operation = 'MAP'

img_size = 28 * 28 # size of MNIST image
img_gray_val = 256 # maximum gray scale of MNIST image
num_class = 10 # number of classes inside MNIST dataset

if_quant = True
position_IM = hdc.item_memory(dim=dim, number=img_size, datatype=datatype)
grayscale_IM = hdc.item_memory(dim=dim, number=img_gray_val, datatype=datatype)     

AM = torch.zeros(size=(num_class, dim), dtype = torch.int32)
position_IM, grayscale_IM, AM = position_IM.to(device), grayscale_IM.to(device), AM.to(device)

hdc.train(AM=AM, train_iter=train_iter, test_iter=test_iter,
      pos_IM=position_IM, val_IM=grayscale_IM, num_classes=num_class,
      if_quant=if_quant, datatype=datatype)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_class).to(device)

for sample, label in track(test_iter, description="Testing"):
    sample, label = sample.to(device), label.to(device)
    hdc.test(accuracy, AM, X_test=sample, Y_test=label,
         pos_IM=position_IM, val_IM=grayscale_IM, if_quant=if_quant, datatype=datatype)
print(f"Dimension {dim}: Test accuracy is {(accuracy.compute().item() * 100):.3f}%")

X, y = next(iter(test_iter))
Y_pred = []
for sample in X:
    Y_pred.append(hdc.predict(AM, sample, position_IM, grayscale_IM, if_quant, datatype=datatype))
Y_pred = torch.tensor(Y_pred)
fig.show_mnist_images(X.reshape(batch_size, 28, 28), 2, 9, titles=preData.get_mnist_labels(Y_pred))