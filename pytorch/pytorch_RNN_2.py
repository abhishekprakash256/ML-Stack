#imports
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#make the tensor for x points
x = torch.linspace(0,799,800)
#make the sign wave
#sin wave is already build in function
y = torch.sin(x*2*3.1416/40)
#print(y)
#plotting the values

#plt.figure(figsize=(20,2))
#plt.title("full_data")
#plt.xlim(-10,801)
#plt.grid(True)
#plt.plot(y.numpy())


#make the train and test split
#plt.title("train_set")
test_size = 40
train_set = y[:-test_size]
test_set = y[-test_size:]

#plt.figure(figsize=(20,2))
#plt.xlim(-10,801)
#plt.grid(True)
#lt.plot(train_set.numpy())
#plt.show()
#make the data for the training
def input_data(seq,ws):
    out = [] #
    L = len(seq)
    for i in range(L-ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window,label))
    return out

window_size = 40

train_data = input_data(train_set,window_size)
#print(len(train_data ))
#make the LSTM model
class LSMT(nn.Module):
    #making the functio with input, hidden state, input_size, output_size
    def __init__(self,input_size = 1,hidden_size = 50,out_size = 1):
        super().__init__()
        self.hidden_size= hidden_size
        self.lstm = nn.LSTM(input_size,hidden_size)
        self.linear = nn.Linear(hidden_size,out_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))

    def forward(self,seq):
        lstm_out , self.hidden =  self.lstm
