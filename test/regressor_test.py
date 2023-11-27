import torch 
from torch import nn 
from torch import optim
from torch.utils.data import  Dataset , DataLoader
from funpredict.torch_train import TrainRegressor
from sklearn.datasets import load_diabetes
data = load_diabetes()
x, y = data.data,data.target

class CustomDataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y 
    def __len__(self):
        return len(self.x)
    def __getitem__(self,idx):
        x_train = torch.from_numpy(self.x[idx])
        y_train = torch.tensor(self.y[idx])
        return x_train.to(torch.float),y_train.to(torch.float) 

dataset = CustomDataset(x,y)
dataloder = DataLoader(dataset,batch_size=10,shuffle=True)

class Net(nn.Module):
    def __init__(self,in_chan=10,out_chan=1):
        super(Net,self).__init__()
        self.li1 = nn.Linear(in_chan,in_chan*2)
        self.li2 = nn.Linear(in_chan*2,in_chan*4)
        self.li3 = nn.Linear(in_chan*4,out_chan)
    def forward(self,x):
        x = self.li1(x)
        x = self.li2(x)
        x = self.li3(x)
        return x

net = Net()
opt = optim.Adam(net.parameters(),lr=0.002) 
loss = nn.L1Loss()

tc = TrainRegressor(net,dataloder,loss,opt)
model = tc.train()
output = model(torch.ones((1,10)))
print(output)
# >>> tensor([[206.1286]])

