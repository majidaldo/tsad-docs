import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



class fig(object):
    def data(self):pass
    def style(self):pass
    def plot(self):pass
    def save(self):pass


class anomtype(fig):
    T=200
    def style(self,po):
        po.axes.get_xaxis().set_ticklabels([])
        po.axes.get_yaxis().set_ticklabels([])
        po.axes.get_xaxis().set_label_text('$t$')
        return po
    def plot(self):
        return self.style(plt.plot(self.data())[0])
    def save(self):
        self.plot().figure.savefig(
            self.__class__.__name__+'.pdf'
            ,bbox_inches='tight'
            ,pad_inches=0
        )

#todo set golden ratio
#figure.savefig('asdfsf.pdf')

class trivial(anomtype):
    def data(self):
        ys=np.random.rand(self.T)
        ys[int(self.T*.5)]=1.5
        return ys

