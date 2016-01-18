import pandas as pd

import tsad
import analysis
import data

import os

#todo def register


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



        
ts=['ecg','sleep','power','spike','sin']

class sampling(tbl):
    
    def get_data(self):
        more=['name','length']
        d=dict.fromkeys(data.get_kwargs(ts[0]).keys())#,[]) <- python gotcha!
        for ak in d: d[ak]=[] #<-soln
        for am in more: d[am]=[]
        for s in ts:
            kw=data.get_kwargs(s)
            kw[more[0]]=s
            kw[more[1]]=data.get_series(s).shape[0]
            for ak in kw: d[ak].append(kw[ak])
        return pd.DataFrame.from_dict(d)
