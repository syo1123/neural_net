import numpy as np
from function import *
class Affine:
    
    def __init__(self,w,b):
        self.w=w
        self.b=b
        self.x=None
        self.original_x_shape=None
        
    def foward(self,x):
        self.original_x_shape=x.shape
        x=x.reshape(x.shape[0],-1)
        self.x=x
        out=np.dot(self.x,self.w)+self.b
        
        return out
    
    def backward(self,dout):
        dx=np.dot(dout,self.w.T)
        self.dw=np.dot(self.x.T,dout)
        self.db=np.sum(dout,axis=0)
        dx=dx.reshape(*self.original_x_shape)
        return dx
    
class Relu:
    
    def __init__(self):
        self.mask=None
        
    def foward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
    
    
class SoftmaxWithLoss:
    def __init__(self):
        self.t=None
        self.y=None
        self.loss=None
        
        
    def foward(self,x,t):
        self.t=t
        self.y=softmax(x)
        self.loss=crros_entropy_error(self.y,self.t)
        return self.loss
    
    def backward(self,dout=1):
        batch_size=self.t.shape[0]
        if self.t.size==self.y.size:
            dx=(self.y-self.t)/batch_size
        else:
            dx=self.y.copy()
            dx[np.arrange(batch_size),self.t]-=1
            dx/batch_size
        return dx
            
        