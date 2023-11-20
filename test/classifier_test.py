import torch 
from torch import nn 
from torch import optim
from torch.utils.data import  Dataset , DataLoader
from funpredict.torch_train import TrainClassifer

x1 = torch.ones((100,1))
x2 = torch.zeros((100,1))
x = torch.cat([x1,x2],dim=0)
y1 = torch.zeros((100))
y2 = torch.ones((100))
y = torch.cat([y1,y2],dim=0)

class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y 
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        x_train = self.x[idx]
        y_train = self.y[idx]
        return x_train,y_train 

dataset = CustomDataset(x,y)
dataloder = DataLoader(dataset,batch_size=10,shuffle=True)

class Net(nn.Module):
    def __init__(self,in_chan=1,out_chan=1):
        super(Net,self).__init__()
        self.li1 = nn.Linear(in_chan,in_chan*2)
        self.li2 = nn.Linear(in_chan*2,in_chan*4)
        self.li3 = nn.Linear(in_chan*4,out_chan)
        self.output = nn.Sigmoid()
    def forward(self,x):
        x = self.li1(x)
        x = self.li2(x)
        x = self.li3(x)
        x = self.output(x)
        return x

net = Net()
opt = optim.Adam(net.parameters(),lr=0.002) 
loss = nn.BCELoss()

tc = TrainClassifer(net,dataloder,loss,opt)
model = tc.train()
print(model)
output = model(torch.ones((1,1)))
print(torch.round(output))

