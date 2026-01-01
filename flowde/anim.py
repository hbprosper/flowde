# ------------------------------------------------------------------------
# GSOC PINNDE Project
# Sijil Jose, Pushpalatha C. Bhat, Sergei Gleyzer, Harrison B. Prosper
# Created: Tue May 27 2025<br>
# Updated: Thu May 29 2025 HBP: generalize sigma(t)
# ------------------------------------------------------------------------
# standard system modules
import os, sys, re

# standard module for array manipulation
import numpy as np

import torch
import torch.nn as nn

import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation

import shutil
# update fonts
plt.rcParams.update({
  "text.usetex": shutil.which('latex') is not None,
  "font.family": "sans-serif",
  "font.sans-serif": "Helvetica",
  "font.size": 14
  })
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
class Flow2DAnimation:
    '''
    Given a Python list, y, of arrays (or tensors), each of shape (N, d) 
    where N is the number of data points and d=2 is the dimensionality of 
    the space, animate the flow of points from a diagonal 2D, zero mean, 
    unit variance Gaussian to the target distribution.

    anim = Flow2DAnimation(y)

    The animation can be saved as a gif or, if ffmpeg is available, in mp4 
    format using

    anim.save('flow.gif')
    anim.save('flow.mp4')

    Parameters
    ----------
        data :      Python list of data objects. Each object is table 
                    (either a 2D numpy array or a 2D tensor) or shape (N, d). 
                    Each data object corresponds to a time step.
        xmin, xmax: x limits [-5, 5]
        ymin, ymax: y limits [-5, 5]
        nframes :   number of frames [50]
        interval :  time between frames in milliseconds [100 ms]
        mcolor :    marker color ['blue']
        msize :     marker size [1]
        fgsize :    size of figure [(4, 4)]
        ftsize :    font size [16 pt]
    '''
    def __init__(self, data,
                 xmin=-5, xmax=5, ymin=-5, ymax=5,
                 nframes=50,     # number of frames
                 interval=100,   # time between frames in milliseconds
                 mcolor='blue',  # color of points
                 msize=1,
                 fgsize=(4, 4), 
                 ftsize=16):
        
        # cache inputs
        self.data = np.array(data)
        self.nframes = nframes
        self.interval = interval
        self.T = len(self.data)
        self.npoints, self.d = self.data[0].shape
        self.factor = self.T / nframes
        
        # set size of figure
        self.fig = plt.figure(figsize=fgsize)

        # create area for a plot 
        nrows, ncols, index = 1, 1, 1
        ax  = plt.subplot(nrows, ncols, index)
        
        self.ax = ax   # cache plot
        
        ax.set_title('Reverse Time Diffusion', pad=14)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        
        # create a scatter plot 
        x = self.data[0, :, 0] # initial x positions
        y = self.data[0, :, 1] # initial y positions
       
        self.scatter = ax.scatter(x, y, s=msize, c=mcolor, alpha=1)

        # matplotlib refers to the objects scatter, etc., as artists.
        # we need to place artists in a list since we need to return all of
        # them in the function update() for the animation to work correctly
        self.artists = []
        self.artists.append(self.scatter)

        # IMPORTANT: Turn off Latex processing of text; it is far too slow!
        mp.rc('text', usetex=False)

        # create a text object to display the days since the start
        self.text = ax.text(0.95*xmin, 0.80*ymin, f't: {0:5.2f}')
        self.artists.append(self.text)

        # fix the layout so that labels are a bit
        # closer to the plot
        self.fig.tight_layout()

        # don't show the above plots. Show only the animated version
        plt.close()

        # initialize animated plot
        self.ani = FuncAnimation(fig=self.fig,           # animation figure
                                 func=self.update,
                                 # function to update plot in figure
                                 repeat=False,
                                 # don't repeat animation
                                 frames=self.nframes,    # number of frames
                                 interval=self.interval)
        # time between frames (ms)

    # this is the function that updates the plot    
    def update(self, frame):
        
        print(f'\rframe: {frame:d}', end='')
        
        # get data and artists!
        data, scatter, text = self.data, self.scatter, self.text
        
        # compute index into data array (i.e., time step)
        index = np.ceil(frame * self.factor).astype(int)
        t = 1 - frame / (self.nframes-1)
        
        # display the number of days
        self.text.set_text(f't: {t:5.2f}')
        
        # display the current position of points
        x = data[index, :, 0]
        y = data[index, :, 1]
        a = np.array([x, y]).T   # we need x and y to be columns
        scatter.set_offsets(a)

        # must return artists for animation to work
        return self.artists
    
    def show(self):
        '''
        Construct and show the animation.
        '''
        plt.show()
        return self.ani
    
    def save(self, filename):
        '''
        Construct and save the animation to a file. The format of the file is
        determined from the file extension. For example, a filename with 
        extension ".gif" saves the animation as an animated gif, while one with
        extension ".mp4" saves it in mp4 format. Note: the latter requires the
        module ffmpeg.
        
    Parameters
    ----------
        filename :      Name of graphics file.
        '''
        self.ani.save(filename)
