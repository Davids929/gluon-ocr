from .recog_dataset import *
from .recog_augment import *
from .detect_dataset import *
from mxnet.gluon.data.vision import transforms

normalize_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])