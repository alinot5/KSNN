#!/usr/bin/env python3
"""
Created on Fri Apr 3 2020

Load KS Hybrid Neural Network (HNN) model for L=22 and display results below
an example trajectory.

Functions:

Trajectory: Takes an initial condition and an amount of time for evolution and 
            outputs the HNN recreation of the trajectory with real time spaced
            every 2 in-slice time units.
__main__: Imports example data, loads the HNN, generates a trajectory and plots.


Important Variables:
    time_units: Amount of time to run the NN
    u0: Initial condition fed into the NN
    stats: [u_mean,u_std,dphi_mean,dphi_std] The state and the change in phase 
            were normalized using the mean and the standard deviation. This was
            done to improve NN training. Normalization was done for both values
            like this: u=(u-u_mean)/u_std.Currently, normalized data is plotted.
    
@author: Alec
"""

# Conda modules
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import ifft
import pickle 
import seaborn as sns

# Local module
import buildmodel_int
    
def Trajectory(u0,time_units,u_std,u_mean):
    
    ut_predict=u0
    tt_pred=np.zeros(1)

    h=2 # in-slice time units     
    # Loop until the time is hit   
    while tt_pred[-1]<time_units:
        # Make a single timestep prediction and append it to the time series
        ut_predict=np.append(ut_predict,model.predict((ut_predict[-1,:])[np.newaxis,:]),axis=0)
        # Get the magnitude of the first Fourier mode
        fmode=abs(ifft(ut_predict[-1,:]*u_std+u_mean)[1])
        # Use the magnitude of the first Fourier mode to go from in-slice time to real time
        tt_pred=np.append(tt_pred,tt_pred[-1]+h*fmode)

    return tt_pred,ut_predict.transpose()

if __name__=='__main__':
    
    ###########################################################################
    # Import Data
    ###########################################################################
    [u,tt,stats,U]=pickle.load(open('Compile_Dat.p','rb'))
    u=(u-stats[0])/stats[1]
    [N,M]=u.shape
    
    ###########################################################################
    # Load Neural Network
    ###########################################################################
    # Build model using keras API
    model,EPOCHS,optimizer=buildmodel_int.buildmodel(N,U,stats[2],stats[3])
    # Compile model
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
    # Load weights of trained model
    model.load_weights('model.h5')
        
    # Print summary of architecture
    print(model.summary())
    
    ###########################################################################
    # Generate an example trajectory
    ###########################################################################
    # Integrate this far forward
    time_units=80
    # Feed in this initial condition
    start=np.random.randint(200,M-4*time_units)
    u0=u[:,start:start+1].transpose()
    # Integrate forward
    ttpred,uts_NN=Trajectory(u0,time_units,stats[1],stats[0])
    
    ###########################################################################
    # Plot
    ###########################################################################

    # Colormap
    colors=sns.diverging_palette(240, 10, n=9,as_cmap=True)
    # Font type and size
    font = {'family' : 'normal','weight' : 'normal','size'   : 10}
    matplotlib.rc('font', **font)
    
    # Create figures
    fig, axs = plt.subplots(2, 1, sharex='col', sharey='row',
                        gridspec_kw={'hspace': .1},figsize=(8/2,4*2/3))
    (ax1), (ax2) = axs
    # Plot data
    im=ax1.pcolormesh(np.arange(0,time_units,.25), np.linspace(-11,11,N), u[:,start:start+int(time_units/.25)], shading='gouraud', cmap=colors,vmin=-3,vmax=3)
    # Plot NN recreation
    last=np.argmin((ttpred-time_units)**2)
    im2=ax2.pcolormesh(ttpred[:last], np.linspace(-11,11,N), uts_NN[:,:last], shading='gouraud', cmap=colors,vmin=-3,vmax=3)
    
    # Label and set limits
    ax1.set_xlim([0,time_units])
    ax1.set(ylabel='x')
    ax2.set(ylabel='x')  
    ax2.set(xlabel='t')

    # Add colorbars
    cax = fig.add_axes([.91, 0.52, 0.02, 0.36]) #left bottom width height
    cb=fig.colorbar(im, cax=cax, orientation='vertical')
    cb.set_label(r'$u$')
    cax2 = fig.add_axes([.91, 0.125, 0.02, 0.36]) #left bottom width height
    cb2=fig.colorbar(im2, cax=cax2, orientation='vertical')
    cb2.set_label(r'$u_{NN}$')
