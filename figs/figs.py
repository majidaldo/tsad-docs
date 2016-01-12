import os

import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from math import sqrt

registry={}
def register(figcls):
    registry[figcls.__name__]=figcls
    return figcls


class fig(object):
    def data(self):pass
    def plot(self): plt.close()
    def style(self):pass
    def format(self):pass
    def save(self):
        self.plot().figure.savefig(self.path())
    def path(self):
#        if not os.path.exists('figs'):
#            os.makedirs('figs')
        pth=os.path.join(''
                         ,self.__class__.__name__+'.pdf'
        )
        return pth

# ts anomaly example figs
    
class oneline(fig):
    def plot(self):
        fig().plot();
        self.format();
        return self.style(plt.plot(self.data())[0]) #[0] b/c jst 1 line

class dualyaxis(fig):
    yl=['y1','y2']
    yc=['b','g']
    yms=['.','.']
    yls=['-','--']; ylsd=['solid','dashed']
    def style(self,po,which):
        plt.setp(po,linewidth=1)
        # po.axes.get_xaxis().set_ticklabels([])
        # po.axes.get_yaxis().set_ticklabels([])
        po.axes.get_xaxis().set_label_text('$t$')
        po.axes.get_yaxis().set_label_text(self.yl[which]
                                           +' ('+self.ylsd[which]+')')
        plt.tight_layout(pad=0)
        return po
    def plot(self):
        fig().plot();
        self.format()
        data=self.data();
        ax2= self.style(   plt.plot(self.data()[0]
                                    ,linestyle=self.yls[0]
                                    ,color=self.yc[0])[0],0 ).axes.twinx()
        return self.style( ax2.plot(self.data()[1]
                                    ,linestyle=self.yls[1]
                                    ,color=self.yc[1])[0],1 )
#import analysis
#import data
class test2(dualyaxis):
    def data(self):
        er=analysis.errs('sin',200)
        ts=data.get_series('sin')[:,0]
#        return ts,er
        return [1,2,3],[5,2,9]

class ts(fig):
    def format(self):
        latexify(6,ratio=.333) #w,r=h*w

    
class anomtype(oneline,ts):#multiple inheritence! i LUV py!
    T=500
    def style(self,po):
        plt.setp(po,linewidth=1)
        po.axes.get_xaxis().set_ticklabels([])
        po.axes.get_yaxis().set_ticklabels([])
        po.axes.get_xaxis().set_label_text('$t$')
        po.axes.get_yaxis().set_label_text('$X$')
        plt.tight_layout(pad=0)
        return po
#todo: vector (small) x

@register
class trivial(anomtype):
    def data(self):
        np.random.seed(123)
        ys=np.random.rand(self.T)
        ys[int(self.T*.5)]=1.5
        return ys

@register
class context(anomtype):
    def data(self):
        ys=np.sin(np.linspace(0,2*np.pi,self.T)*8)
        ys[int(self.T*.5)]=.75
        return ys

    
def gaussian(x, mu, sig):
    n=-np.power(x - mu, 2.)
    d=(2 * np.power(sig, 2.))
    return np.exp(n/d)
    
@register
class discord_per(anomtype):#_periodic
    def data(self):
        ys=np.sin(np.linspace(0,2*np.pi,self.T)*8)
        mp=gaussian(np.linspace(-1,1,self.T),0,.1)
        return np.multiply((-.5*mp+1),ys)

@register
class discord_aper(anomtype):
    def data(self):
        def g(l,s=.01):
            return gaussian(np.linspace(0,1,self.T),l,s)
        gs=g(-999)
        for al in [0.025,.1,.3,.6,.7,.9]: gs+=g(al)
        return gs+g(.5,s=.0025)

    
# clustering figs

class xy(fig):
    def plot(self,**kwargs):
        #fig().plot()
        self.format()
        return self.style(plt.plot(*self.data(),**kwargs)[0])
    

def clusterdata(x,y,n=10,cv=[[.008,0],[0,0.008]]):
    mn=[x,y]
    np.random.seed(1999)
    return np.random.multivariate_normal(mn,cv,n).T


class cluster(xy):
    def style(self,po,**kwargs):
        plt.setp(po,linestyle='none',markersize=4,**kwargs)
        po.axes.get_xaxis().set_ticklabels([])
        po.axes.get_yaxis().set_ticklabels([])
        plt.tight_layout(pad=0)
        return po  
    def format(self):
        latexify(2,ratio=1) #w,r=h*w


class densell(cluster):
    def data(self): return clusterdata(.25,.25,30)
class apt(cluster):
    def data(self): return [[.25],[.8]]

class clusters(cluster):
    clusters=None
    def plot(self):
        for acc,stl,nt in self.clusters:
            co=acc()
            p=co.plot(**stl)
            xy=np.array((np.median(p.get_xdata())
                         ,.0+np.max(p.get_ydata())))
            plt.annotate(nt,xy
                         ,xytext=(10,0)
                         ,textcoords='offset points')
        return p

@register
class simple_dist(clusters):
    clusters=[(densell
               ,{'color':'darkblue','marker':'o'}
               ,'$\mathcal{N}_1$')
              ,(apt
                ,{'color':'darkblue','marker':'^'}
                ,'$p_1$')]
    
class apt2(cluster):
    def data(self): return [[.27],[.7]]


@register
class hard1_dist(clusters):
    clusters=simple_dist.clusters[:]\
              +[(apt2
                 ,{'color':'darkblue','marker':'^'}
                 ,'$p_2$')]

class sparse(cluster):
    def data(self):
        return clusterdata(1.5,2
                           ,15
                           ,[[.2,0],[0,.2]])

@register
class hard2_dist(clusters):
    clusters=hard1_dist.clusters[:]\
              +[(sparse
                 ,{'color':'darkblue','marker':'s'}
                 ,'$\mathcal{N}_2$')]
    
#----    
def latexify(fig_width=None
             , fig_height=None
             ,ratio='golden'
             , columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert(columns in [1,2])

    if fig_width is None:
        fig_width = 3.39 if columns==1 else 6.9 # width in inches

    if fig_height is None:
        if ratio=='golden':
            ar = (sqrt(5)-1.0)/2.0    # Aesthetic ratio
        else:
            ar=ratio
            fig_height = fig_width*ar # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + str(fig_height) + 
              "so will reduce to" + str(MAX_HEIGHT_INCHES) + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {'backend': 'ps',
              'text.latex.preamble': ['\usepackage{gensymb}'],
              'axes.labelsize': 10, # fontsize for x and y labels (was 10)
              'axes.titlesize': 10,
              'font.size': 10, # was 10
              'legend.fontsize': 10, # was 10
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'text.usetex': True,
              'figure.figsize': [fig_width,fig_height],
              'font.family': 'serif'
    }

    matplotlib.rcParams.update(params)


def format_axes(ax):

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


if __name__=='__main__':
    import sys
    fignm=sys.argv[1]
    
    if fignm=='all': fignm=registry.keys()
    else: fignm=[fignm]
    
    for afn in fignm: plt.close(); eval(afn+'().save()');
