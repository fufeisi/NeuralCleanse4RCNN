import torch, torchvision
from torchvision import datasets
from logging import Logger
from defense.neural_cleanse import NeuralCleanse

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1*28*28, 10), torch.nn.ReLU())
model.num_classes = 10
data = datasets.MNIST('/Users/feisifu/VSProjects/data', train=False, transform=torchvision.transforms.ToTensor())
input_shape = [1, 28, 28]
log_path = 'log'
logger = Logger('my_logger')
device = 'cpu'

my_cleanse = NeuralCleanse(model, input_shape, data, log_path, logger, device=device)
my_cleanse.detect()
