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


transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))]
)
cifar_trainset = datasets.CIFAR10(root = "./CIFAR10", train = True, download= False, transform= transforms)
cifar_testset = datasets.CIFAR10(root = "./CIFAR10", train = False, download= False, transform= transforms)

train_loader = DataLoader(dataset= cifar_trainset, batch_size=32, shuffle= True, num_workers= 2)
test_loader = DataLoader(dataset= cifar_testset, batch_size=32, shuffle= False, num_workers= 2)

class SimpleCNN(nn.Module):
    ''' 
    Simple CNN for a classification Problem
    '''
    def __init__(self, input_channels = 3, num_classes = 10):
        super(SimpleCNN,self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels= self.input_channels, out_channels=32, kernel_size= 5, stride= 2, padding=2)
        self.conv2 = nn.Conv2d(in_channels= self.conv1.out_channels, out_channels=64, kernel_size= 3, stride= 2, padding= 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dense = nn.Linear(in_features= self.conv2.out_channels, out_features= self.num_classes )
    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        x = self.avgpool(x)
        out = self.dense(torch.reshape(x, (x.shape[0],self.conv2.out_channels)))
        out = F.softmax(out, dim=1)
        return out

model = SimpleCNN()
model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters(),lr = 0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metrics = {
    "accuracy": Accuracy(),
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
    print(f"Training Epoch: {train_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} Avg Acc: {metrics['accuracy']:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def loss_validator_results(trainer):
    test_evaluator.run(test_loader)
    metrics = test_evaluator.state.metrics
    print(f"Validation Epoch: {test_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} Avg Acc: {metrics['accuracy']:.2f}")

def score_function(engine):
    return engine.state.metrics["accuracy"]

model_checkpoint = ModelCheckpoint(
    "checkpoint_CIFRA10",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)
  
test_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

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
