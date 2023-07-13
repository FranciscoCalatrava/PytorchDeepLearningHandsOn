import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

@dataclass(eq = False)
class FizzBuzzDataset(Dataset):
    input_size: int = 10
    start: int = 0
    end: int = 1000

    def encoder(self, num):
        ret = [int(i) for i in '{0:b}'.format(num)]
        return [0]*(self.input_size-len(ret))+ret
    
    def __getitem__(self, index):
        index += self.start
        x = self.encoder(index)
        if index % 15 == 0:
            y = 0
        if index % 5 == 0:
            y = 1
        if index % 3 == 0:
            y = 2
        else:
            y = 3
        return torch.tensor(x).float(), torch.tensor(y)
    def __len__(self):
        return self.end-self.start
    
class FizzBuzzNetwork(nn.Module):
    """
    Network for the game
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(FizzBuzzNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
    def forward(self, input):
        x = F.relu(self.hidden(input))
        out = self.output(x)
        return out
    
##We define the dataset and we split it into train and validation

dataset = FizzBuzzDataset()
dataset_train, dataset_val = random_split( dataset, [800, 200]) 

##We create the dataloader

train_loader = DataLoader(dataset= dataset_train, 
                          batch_size= 10, 
                          shuffle= True)

val_loader = DataLoader(dataset= dataset_val, 
                          batch_size= 10, 
                          shuffle= True)

## Let's define the model and the loss function

model =  FizzBuzzNetwork(input_size = 10, hidden_size = 100, output_size = 4)
model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Now we have to define all the things from  pytorch ignite

trainer = create_supervised_trainer(model= model, optimizer= optimizer, loss_fn= loss_fn, device= device)

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(loss_fn)
}


train_evaluator = create_supervised_evaluator(model= model, metrics= metrics, device= device)
val_evaluator = create_supervised_evaluator(model, metrics= metrics, device= device)


## Now we define the function from pytorch-ignite for the metrics
log_interval = 100


@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def loss_trainer(engine):
    print(f"Epoch: {engine.state.epoch} Iteration: {engine.state.iteration:.2f} Loss: {engine.state.output:.2f}")

@trainer.on(Events.EPOCH_COMPLETED)
def loss_trainer_results(trainer):
    train_evaluator.run(train_loader)
    metrics = train_evaluator.state.metrics
    print(f"Training Epoch: {train_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} Avg Acc: {metrics['accuracy']:.2f}")
    
@trainer.on(Events.EPOCH_COMPLETED)
def loss_validator_results(trainer):
    val_evaluator.run(val_loader)
    metrics = train_evaluator.state.metrics
    print(f"Validation Epoch: {val_evaluator.state.epoch:.2f} Avg Loss: {metrics['loss']:.2f} Avg Acc: {metrics['accuracy']:.2f}")

def score_function(engine):
    return engine.state.metrics["accuracy"]


model_checkpoint = ModelCheckpoint(
    "checkpoint",
    n_saved=2,
    filename_prefix="best",
    score_function=score_function,
    score_name="accuracy",
    global_step_transform=global_step_from_engine(trainer),
)
  
val_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})

tb_logger = TensorboardLogger(log_dir="tb-logger")

tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=100),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

for tag, evaluator in [("training", train_evaluator), ("validation", val_evaluator)]:
    tb_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag=tag,
        metric_names="all",
        global_step_transform=global_step_from_engine(trainer),
    )


trainer.run(train_loader, max_epochs=3000)
