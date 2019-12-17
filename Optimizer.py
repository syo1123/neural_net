import numpy as np
class Momentum:
    def __init__(self,momentum=0.9):
        
        self.momentum=momentum
        self.velocity={}
        
    def up(self,params,grad,learning_rate):
        if len(self.velocity)==0:
            for key,val in params.items():
                self.velocity[key]=np.zeros_like(val)
        for key in params.keys():
            self.velocity[key]=self.momentum*self.velocity[key]-learning_rate*grad[key]
            params[key]+=self.velocity[key]
            
        return params

class Sgd:
	def __init__(self):
		self.sgd=None
		
	def up(self,params,grad,learning_rate):
		for key in params.keys():
			params[key]-=learning_rate*grad[key]
		return params
		
		
class AdaGrad:
	def __init__(self):
		self.d={}
		
	def up(self,params,grad,learning_rate):
		if len(self.d)==0:
			for key,val in params.items():
				self.d[key]=np.zeros_like(val)
		for key in params.keys():
			
			self.d[key]+=grad[key]**2
			params[key]-=learning_rate*grad[key]/(np.sqrt(self.d[key]))
		return params