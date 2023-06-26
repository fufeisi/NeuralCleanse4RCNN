import torch, torchvision
from torchvision import datasets
from logging import Logger
from defense.neural_cleanse import NeuralCleanse

model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(1*28*28, 10), torch.nn.ReLU())
model.num_classes = 10
data = zip([torch.rand([64, 1, 28, 28])], [torch.randint(0, 9, [64])])
input_shape = [1, 28, 28]
log_path = 'log'
logger = Logger('my_logger')
device = 'cuda'

my_cleanse = NeuralCleanse(model, input_shape, data, log_path, logger, device=device)
my_cleanse.detect()
