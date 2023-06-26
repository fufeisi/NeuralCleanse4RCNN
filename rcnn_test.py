import torch, torchvision
from torchvision import datasets
from logging import Logger
from defense.neural_cleanse import NeuralCleanse

# fake data
batch = 4
images, boxes = torch.rand(batch, 3, 600, 1200), torch.rand(batch, 11, 4)
boxes[:, :, 2:4] = boxes[:, :, 0:2] + boxes[:, :, 2:4]
labels = torch.randint(1, 91, (batch, 11))
targets = []
for i in range(len(images)):
   d = {}
   d['boxes'] = boxes[i]
   d['labels'] = labels[i]
   targets.append(d)

model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
model.num_classes = 90
data = zip([images], [targets])
input_shape = [3, 600, 1200]
log_path = 'log'
logger = Logger('my_logger')
device = 'cuda'

my_cleanse = NeuralCleanse(model, input_shape, data, log_path, logger, device=device, is_rcnn=True)
my_cleanse.detect()
