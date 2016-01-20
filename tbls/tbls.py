import pandas as pd

import tsad
import analysis
import data

import os

# todo def register
registry=[]
def register(cls): registry.append(cls)

class tbl(object):
    kwargs={};
    def kwargsproc(self,kwargs): return kwargs
#    self __init__(self,*args,**kwargs): self.data=self.get_data()
    def get_data(self): pass
    def todf(self,data): return pd.DataFrame(data)
    def save(self,*args,**kwargs):
        df=self.todf(self.get_data())
        kwargs=self.kwargsproc(kwargs)
        df.to_csv(path_or_buf=os.path.join(os.getcwd()
                                           ,self.__class__.__name__+'.csv')
                  ,index=False
                  ,**kwargs)

       
ts=[
    'spike',
    'sin',
    'power',
    'ecg',
    'sleep',
]


@register
class sampling(tbl):
    
    def get_data(self):
        more=['name','length','nsamples']
        d=dict.fromkeys(data.get_kwargs(ts[0]).keys())#,[]) <- python gotcha!
        for ak in d: d[ak]=[] #<-soln
        for am in more: d[am]=[]
        for s in ts:
            kw=data.get_kwargs(s)
            kw[more[0]]=s
            kw[more[1]]=data.get_series(s).shape[0]
            kw[more[2]]=len(data.get(s))
            for ak in kw: d[ak].append(kw[ak])
        return pd.DataFrame.from_dict(d)




    
#if __name__=='__main__':
#    for ac in registry: cls().save()
