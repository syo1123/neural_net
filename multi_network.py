


import sys,os
sys.path.append(os.pardir)
from layer import *
from collections import OrderedDict

class network:
    
    def __init__(self,input_size,hidden_size,hidden_layer_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['w1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        for i in range(hidden_layer_size):
            self.params['w'+str(i+2)]=weight_init_std*np.random.randn(hidden_size,hidden_size)
            self.params['b'+str(i+2)]=np.zeros(hidden_size)
        self.params['w'+str(hidden_layer_size+2)]=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b'+str(hidden_layer_size+2)]=np.zeros(output_size)
        
        self.layers=OrderedDict()
        for i in range(hidden_layer_size+2):
            self.layers['Affine'+str(i+1)]=Affine(self.params['w'+str(i+1)],self.params['b'+str(i+1)])
            if hidden_layer_size+2-i==1:break
            self.layers['Relu'+str(i+1)]=Relu()
        self.last_layer=SoftmaxWithLoss()
        
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.foward(x)
        return x
    
    def loss(self,x,t):
        y=self.predict(x)
        return self.last_layer.foward(y,t)
    
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        if t.ndim !=1:t=np.argmax(t,axis=1)
        
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
        
    
    def grad(self,x,t):
        self.loss(x,t)
        
        dout=1
        dout=self.last_layer.backward(dout)
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        
        grad={}
        grad['w1'],grad['b1']=self.layers['Affine1'].dw,self.layers['Affine1'].db
        grad['w2'],grad['b2']=self.layers['Affine2'].dw,self.layers['Affine2'].db
        grad['w3'],grad['b3']=self.layers['Affine3'].dw,self.layers['Affine3'].db
        
        return grad

