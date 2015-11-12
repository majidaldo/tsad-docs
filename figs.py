import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from math import sqrt

class fig(object):
    def data(self):pass
    def style(self):pass
    def plot(self):pass
    def format(self):pass
    def save(self):pass


class anomtype(fig):
    T=200
    def style(self,po):
        po.axes.get_xaxis().set_ticklabels([])
        po.axes.get_yaxis().set_ticklabels([])
        po.axes.get_xaxis().set_label_text('$t$')
        plt.tight_layout(pad=0)
        return po
    def format(self):
        latexify(6,ratio=.3) #w,r=h*w
    def plot(self):
        self.format()
        return self.style(plt.plot(self.data())[0])
    def save(self):
        self.plot().figure.savefig(
            self.__class__.__name__+'.pdf'
        )


class trivial(anomtype):
    def data(self):
        ys=np.random.rand(self.T)
        ys[int(self.T*.5)]=1.5
        return ys



    
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
              'text.fontsize': 10, # was 10
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
    eval(fignm+'().save()')
