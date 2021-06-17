import torch
import progressbar
import logging

## Model Trainer Class
"""
This class is a multi-purpose PyTorch model trainer class designed
to handle the training and testing iterations of a model
args:
    model: nn.Module model to train
    loss_fn: loss function to train with
    optimizer: optimizer function
    device: device the model is on (cpu or cuda)
"""
class Trainer:
    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.plan = []

    # get model after training
    def GetModel(self):
        return self.model

    # helper to add train/test to training plan
    def _addWorkout(self, workout):
        workout = workout.lower()
        if workout == 'train' or workout == 'tr':
            self.plan.append('train')
        elif workout == 'test' or workout == 'te':
            self.plan.append('test')
    
    # generate a model "training plan"
    """
        create a plan consisting of train/test epochs to run
        expects a list of strings and ints to generate the plan, 
        "tr" = "train", "te" = "test"
        use an int to run the previous label that many times
        ["train", 5, "test"] will run 5 training epochs followed by a testing epoch
    """
    def SetTrainingPlan(self, training_plan):
        self.plan = []

        for i, workout in enumerate(training_plan):
            if type(workout) == int:
                for j in range(workout-1):
                    self._addWorkout(training_plan[i-1])
            else:
                self._addWorkout(workout)

    def SetEpochs(self, num_epochs):
        self.plan = []

        for i in range(num_epochs):
            self._addWorkout("train")
            self._addWorkout("test")

    def Run(self, train_dataloader, test_dataloader = None, verbose = True):
        logger=logging.getLogger()
        # TODO: print training plan

        for i, w in enumerate(self.plan):
            if w == 'train':
                logger.info(f"training, epoch {i+1} of {len(self.plan)}")
                self._train(train_dataloader, verbose)

            elif w == 'test':
                if not test_dataloader:
                    print("oops") # TODO: handle error

                logger.info(f"testing, epoch {i+1} of {len(self.plan)}")
                self._test(test_dataloader, verbose)

    # private train function, provides one epoch of training
    def _train(self, dataloader, verbose):
        size = len(dataloader.dataset)

        # logger and progress bar setup
        logger=logging.getLogger()

        widgets=[
          f"training...",
          progressbar.Bar(), 
          progressbar.Percentage()  
        ]

        if verbose: bar = progressbar.ProgressBar(max_value=size, widgets=widgets)

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device) # move feature and label to device

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # log current loss 10 times through epoch
            if batch % int(size/len(X)/10) == 0:
                loss, current = loss.item(), batch * len(X)
                logger.info(f"  loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

            if verbose: bar.update(batch*len(X)) # iterate progressbar
        if verbose: bar.finish()

    # private test function
    def _test(self, dataloader, verbose):
        size = len(dataloader.dataset)

        # progressbar setup
        widgets=[
          f"testing... ",
          progressbar.Bar(), 
          progressbar.Percentage()  
        ]

        if verbose: bar = progressbar.ProgressBar(max_value=size, widgets=widgets)

        self.model.eval() # set model to eval mode
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                if verbose: bar.update(batch*len(X))
        if verbose: bar.finish()

        logging.getLogger().info(f"test accuracy: {(100*correct/size):>0.1f}%, avg loss: {test_loss/size:>8f}")