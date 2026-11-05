import numpy as np
import torch
import datetime
import os

class EarlyStopping(object):
    def __init__(self, patience=10, save_path='checkpoint.pt'):
        dt = datetime.datetime.now()
        os.makedirs('checkpoint', exist_ok=True)
        self.filename = 'checkpoint/{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False
        self.save_path = save_path
        #print(self.save_path)

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        # elif acc <= self.best_acc:
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # if (loss <= self.best_loss) and (acc >= self.best_acc):
            if acc >= self.best_acc:
                self.save_checkpoint(model)
                print("Model state saved!!!")
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)
        # print("Saves model!")

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
