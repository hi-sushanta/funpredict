import torch 
from torch import nn
from tqdm import tqdm 
from torch import optim
import matplotlib.pyplot as plt
class TrainClassifer:
    def __init__(self,model,dataloader,loss,opt,lr=0.002,device='cpu',):
        self.model  = model 
        self.dataloader = dataloader
        self.lr = lr
        self.opt = opt
        self.loss = loss
        self.tracking_loss = []
        self.tracking_ep = []
        self.device=device
        self.fig = None 
        self.ax = None
    def train(self,epoch=100):
        self.fig=None 
        self.ax = None
        for i in range(epoch):
            total_loss = []
            for item in tqdm(self.dataloader):
                x = item[0].to(self.device)
                y = item[1].to(self.device)
                self.model.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred.squeeze(),y)
                total_loss.append(loss.item())
                loss.backward()
                self.opt.step()
            self.tracking_loss.append(sum(total_loss)/len(self.dataloader))
            self.tracking_ep.append(i+1)
            self.loss_plot(i,self.tracking_loss[i])
        return self.model
    
    def loss_plot(self,epoch:int,loss:float,sign='o'):
         # Create a new figure if needed
        if self.fig is None:
            self.fig, self.ax = plt.subplots()

        # Add the current loss to the plot
        self.fig.suptitle("Model Tracking")
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.plot(epoch, loss, sign)

        # Update the plot
        plt.draw()
        plt.pause(0.01)


    


