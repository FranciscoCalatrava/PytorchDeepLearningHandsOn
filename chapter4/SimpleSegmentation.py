import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from fastcore.all import *
import torchdata.datapipes.iter as pipes
from torch.utils.data.datapipes.utils.decoder import imagehandler
from torchdata.datapipes.iter import FileOpener, IterableWrapper, Mapper, RoutedDecoder, TarArchiveLoader

from torch.utils.data import Dataset
import cv2
import torch
import os
from imutils import paths
from sklearn.model_selection import train_test_split

class convblock(nn.Module):
    ''' 
    Conv block conv -> BN -> RELU
    '''
    def __init__(self, input_channel, output_channel, kernel, padding, stride, bias):
        super(convblock, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel = kernel
        self.padding = padding
        self.stride = stride
        self.bias = bias
        self.conv = nn.Conv2d(in_channels = self.input_channel, out_channels = self.output_channel, kernel_size= self.kernel, padding= self.padding, stride= self.stride, bias= self.bias)
        self.bn = nn.BatchNorm2d(num_features= self.output_channel)
        self.relu = nn.ReLU()
    
    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        out = self.relu(x)
        return out

class deconvblock(nn.Module):
    ''' 
    deconvolution block 
    '''
    def __init__(self, input_channel, output_channel, kernel, padding, stride, bias):
        super(deconvblock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels= input_channel, out_channels= output_channel, kernel_size= kernel, stride= stride, padding= padding)
        self.bn = nn.BatchNorm2d(num_features= output_channel)
        self.relu = nn.ReLU()
    def forward(self, input, output_size):
        x = self.deconv(input, output_size = output_size)
        x = self.bn(x)
        out = self.relu(x)
        return out

class EncoderBlock(nn.Module):
    '''
    Encoder block from the LinkNet for semantic segmentation
    '''
    def __init__(self, output_channel, input_channel = 3):
        super(EncoderBlock, self).__init__()
        self.input_channel = 3
        self.convblock1 = convblock(input_channel=input_channel, output_channel= output_channel, kernel= 3, stride= 2, padding= 1, bias= True)
        self.convblock2 = convblock(input_channel= self.convblock1.output_channel, output_channel= output_channel, kernel= 3, stride= 1, padding= 1, bias= True)
        self.convblock3 = convblock(input_channel= self.convblock1.output_channel, output_channel= output_channel, kernel= 3, stride= 1, padding= 1, bias= True)
        self.convblock4 = convblock(input_channel= self.convblock1.output_channel, output_channel= output_channel, kernel= 3, stride= 1, padding= 1, bias= True)
        self.residue = convblock(input_channel=input_channel, output_channel= output_channel, kernel= 3, stride= 2, padding= 1, bias= True)
    def forward(self, input):
        input_1 = input
        x = self.convblock1(input)
        x = self.convblock2(x)
        input_2 = self.residue(input_1) + x
        x = self.convblock3(input_2)
        x = self.convblock4(x)
        out = x + input_2
        return out
    

class DecoderBlock(nn.Module):
    ''' 
    Decoder block from the LinkNet for semantic segmentation
    '''

    def __init__(self, input_channel, output_channel):
        super(DecoderBlock, self).__init__()
        self.input_channel = 3
        self.conv1 = convblock(input_channel= input_channel, output_channel= input_channel//4, kernel= 1, stride= 1,  padding= 0, bias= True )
        self.deconv = deconvblock(input_channel= input_channel//4, output_channel= input_channel//4, kernel= 3, stride= 2, padding= 1, bias=True)
        self.conv2 = convblock(input_channel= input_channel//4, output_channel= output_channel, kernel= 1, stride= 1, padding=0, bias= True)

    def forward(self, input, output_size):
        x = self.conv1(input)
        x = self.deconv(x, output_size = output_size)
        x = self.conv2(x)
        return x
    
class SegmentationModel(nn.Module):
    def __init__(self,):
        super(SegmentationModel, self).__init__()
        self.conv1 = convblock(input_channel= 3, output_channel= 64, kernel= 7, stride= 2, padding= 3, bias= True)
        self.maxpool = nn.MaxPool2d(kernel_size= 3, stride= 2, padding= 1)
        self.encoder1 = EncoderBlock(input_channel= 64, output_channel=64)
        self.encoder2 = EncoderBlock(input_channel= 64, output_channel=128)
        self.encoder3 = EncoderBlock(input_channel= 128, output_channel=256)
        self.encoder4 = EncoderBlock(input_channel= 256, output_channel=512)
        self.decoder1 = DecoderBlock(input_channel=512, output_channel=256)
        self.decoder2 = DecoderBlock(input_channel=256, output_channel=128)
        self.decoder3 = DecoderBlock(input_channel=128, output_channel=64)
        self.decoder4 = DecoderBlock(input_channel=64, output_channel=64)

        self.final_deconv = deconvblock(input_channel= 64, output_channel= 32, kernel=3, stride= 2, padding= 1, bias=True)
        self.final_conv = convblock(input_channel=32, output_channel=32, kernel=3, stride=1, padding=1, bias=True)
        self.final_deconv_1 = deconvblock(input_channel=32, output_channel=1, kernel=2, stride=2, padding=0, bias=True)

        self.init_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self, input):
        init_conv = self.conv1(input)
        init_maxpool = self.maxpool(init_conv)
        x_1 = self.encoder1(init_maxpool)
        x_2 = self.encoder2(x_1)
        x_3 = self.encoder3(x_2)
        x_4 = self.encoder4(x_3)
        x_5 = self.decoder1(x_4, x_3.size()) + x_3
        x_6 = self.decoder2(x_5, x_2.size()) + x_2
        x_7 = self.decoder3(x_6, x_1.size()) + x_1
        x_8 = self.decoder4(x_7, init_maxpool.size())
        out = self.final_deconv(x_8, init_conv.size())
        out = self.final_conv(out)
        out = self.final_deconv_1(out, input.size())
        return out

class SegmentationDataset(Dataset):
    def __init__(self, imgpath, maskpath, transform):
        self.imgpath = imgpath
        self.maskpath = maskpath
        self.transform  = transform
    def __len__(self):
        return len(self.imgpath)
    
    def __getitem__(self, index):
        imagePath = self.imgpath[index]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskpath[index], 0 )

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return (image, mask)

# base path of the dataset
DATASET_PATH = os.path.join("segmentation", "competition_data", "train")
# define the path to the images and masks dataset
IMAGE_DATASET_PATH = os.path.join(DATASET_PATH, "images")
MASK_DATASET_PATH = os.path.join(DATASET_PATH, "masks")
# define the test split
TEST_SPLIT = 0.15
# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# define the number of channels in the input, number of classes,
# and number of levels in the U-Net model
NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3
# initialize learning rate, number of epochs to train for, and the
# batch size
NUM_EPOCHS = 40
BATCH_SIZE = 64
# define the input image dimensions
INPUT_IMAGE_WIDTH = 128
INPUT_IMAGE_HEIGHT = 128
# define threshold to filter weak predictions
THRESHOLD = 0.5


imagePaths = sorted(list(paths.list_images(IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(MASK_DATASET_PATH)))

split = train_test_split(imagePaths, maskPaths, test_size= TEST_SPLIT, random_state=42)

(trainImages, testImages) = split[:2]
(trainMask, testMask) = split[2:]

transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH)), transforms.ToTensor()])

trainDS = SegmentationDataset(imgpath= trainImages, maskpath= trainMask, transform= transforms)
testDS = SegmentationDataset(imgpath= testImages, maskpath = testMask, transform= transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")

train_loader = DataLoader(trainDS, shuffle= True, batch_size= BATCH_SIZE, pin_memory = PIN_MEMORY,  num_workers= os.cpu_count())
test_loader = DataLoader(testDS, shuffle= False, batch_size= BATCH_SIZE, pin_memory= PIN_MEMORY, num_workers= os.cpu_count())
model = SegmentationModel()
model.cuda()
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(params= model.parameters(),lr = 0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = {
    "loss": Loss(loss_fn)
}

trainer = create_supervised_trainer(model= model, optimizer= optimizer, loss_fn = loss_fn, device= device)
train_evaluator = create_supervised_evaluator(model= model, metrics= metrics, device = device)
test_evaluator = create_supervised_evaluator(model= model, metrics= metrics, device = device)

log_interval = 100


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def loss_training(engine):
    print(f"Epoch {engine.state.epoch} Iteration {engine.state.iteration:.2f} Loss: {engine.state.output:2f}")


@trainer.on(Events.EPOCH_COMPLETED)
def trainer_avg(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Epoch: {train_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} ")

@trainer.on(Events.EPOCH_COMPLETED)
def loss_validator_results(trainer):
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    print(f"Validation Epoch: {test_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} ")




tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", test_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


trainer.run(train_loader, max_epochs=200)
