import torch, torchvision
from torchvision import datasets
from logging import Logger
from defense.neural_cleanse import NeuralCleanse

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.num_classes = 10
data = datasets.CIFAR10('/Users/feisifu/VSProjects/data', train=False, transform=torchvision.transforms.ToTensor())
input_shape = [3, 32, 32]
log_path = 'log'
logger = Logger('my_logger')
device = 'cpu'

my_cleanse = NeuralCleanse(model, input_shape, data, log_path, logger, device=device, is_rcnn=True)
my_cleanse.detect()
