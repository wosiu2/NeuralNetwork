# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 22:01:46 2018

@author: Michał
"""

import torch
import torch.nn as nn



class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,classes):
        super(NeuralNet,self).__init__()
        self.layerIn=nn.Linear(input_size,hidden_size)
        self.relu1=nn.ReLU()
        self.layerInter=nn.Linear(hidden_size,hidden_size)
        self.relu2=nn.ReLU()
        self.layerOut=nn.Linear(hidden_size,classes)
        
    def forward(self,x):
        out=self.layerIn(x)
        out=self.relu1(out)
        out=self.layerInter(out)
        out=self.relu2(out)
        out=self.layerOut(out)
        return out
    



class NeuralModule():
    def __init__(self, nnet):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate=0.001
        self.model=nnet.to(self.device)
        self.criterion=nn.MSELoss(reduction='sum')#nn.CrossEntropyLoss()
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.learning_rate)
        self.epochs=1000
    
    def SingleTrain(self,inData,outData):
        input_data=inData.to(self.device)
        output_data=outData.to(self.device)
    
        outputs=self.model(input_data)
        loss=self.criterion(outputs,output_data)
       
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
     
    def Train(self,inData,outData):
        
        for i in range(self.epochs):
            self.SingleTrain(inData,outData)
            
    def Result(self,inData):
        
        return self.model(inData)
    

if __name__=="__main__":
    #Parameters

    input_size=5
    hidden_size=100
    classes=2
    
   
    learning_rate=0.0001

    model=NeuralNet(input_size,hidden_size,classes)
    module=NeuralModule(model)
    print(module.device)
    
    i=torch.randn(20,5)
    j=torch.randn(20,2)
    
    module.Train(i,j)
    
    print(j)
    print(module.Result(i))

