#!/usr/bin/env python3
'''

Created on Fri Apr 3 2020

This contains the architecture for the NN.

Functions:
    
trunc: Outputs the size of the reduced dimension.
Shift: Phase aligns data using the method of slices.
Shift_Back: Takes in-slice data and shifts it back to the fullspace
buildmodel: Uses keras API to create a model with the parameters specified at 
            the beginning of the function.

@author: Alec
'''
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
import tensorflow.keras.backend as K

def trunc():

    return 7

def Shift(dense):
    N=64
    # Get the phase
    utfft=tf.ifft(tf.cast(dense,dtype=tf.complex128))
    utfft_real=tf.math.real(tf.slice(utfft,[0,1],[-1,1]))
    utfft_imag=tf.math.imag(tf.slice(utfft,[0,1],[-1,1]))
    phase=tf.atan2(utfft_imag,utfft_real)

    # Shift the data
    utfft=tf.fft(tf.cast(dense,dtype=tf.complex128))
    seq1=tf.lin_space(0.0,N/2-1.0,int(N/2))
    seq2=tf.lin_space(-N/2,-1.0,int(N/2))
    k=tf.concat([seq1,seq2],0)
    k=tf.cast(k,dtype=tf.complex128)
    exp=tf.math.exp(1j*tf.linalg.matmul(tf.cast(phase,dtype=tf.complex128),tf.expand_dims(k,0)))
    utfft=tf.math.multiply(utfft,exp)
    
    main_output=tf.cast(tf.ifft(tf.cast(utfft,dtype=tf.complex128)),dtype=tf.float64)
    main_output=tf.concat([main_output,phase],1)

    return main_output 

def Shift_Back(dense):
    N=64
    # Get the phase
    phase=tf.slice(dense,[0,N],[-1,1])
    dense=tf.slice(dense,[0,0],[-1,N])
    
    # Shift the data
    utfft=tf.fft(tf.cast(dense,dtype=tf.complex128))
    seq1=tf.lin_space(0.0,N/2-1.0,int(N/2))
    seq2=tf.lin_space(-N/2,-1.0,int(N/2))
    k=tf.concat([seq1,seq2],0)
    k=tf.cast(k,dtype=tf.complex128)
    exp=tf.math.exp(-1j*tf.linalg.matmul(tf.cast(phase,dtype=tf.complex128),tf.expand_dims(k,0)))
    utfft=tf.math.multiply(utfft,exp)
    
    main_output=tf.cast(tf.ifft(tf.cast(utfft,dtype=tf.complex128)),dtype=tf.float64)

    return main_output 

# Dense layer equivariance in main function
def buildmodel(N,U,dtheta_mean,dtheta_std):
    # Set precision
    K.set_floatx('float64')
    
    # PCA change of basis matrix
    Utens=K.variable(value=U)
    
    # Tuning parameters
    optimizer = tf.train.AdamOptimizer(0.0001)
    EPOCHS=200

    ########################################################################################
    # NN Structure
    ########################################################################################
    # State Encoder
    hiddenin=[500,trunc()]
    actin=['sigmoid','tanh']

    # State Decoder
    hiddenout_PCA=[hiddenin[0],trunc()]
    hiddenout=[hiddenin[0],N-trunc()]
    actout=['sigmoid',None]

    # State Evolution
    act_step=['sigmoid','sigmoid',None]
    timestep=[200,200,trunc()]   
    
    # Phase Evolution
    hidden_phase=[500,50,500,1] 
    act_phase=['sigmoid','sigmoid','sigmoid',None]
    
    ########################################################################################
    # Encoder
    ########################################################################################
    # Input size
    main_input = layers.Input(shape=(N,), name='main_input')

    # Remove phase
    shift_input=Lambda(Shift,name='MoS_In')(main_input)
    phase_input=Lambda(lambda x: tf.slice(x,[0,N],[-1,1]),name='Get_Phase')(shift_input)
    shift_input=Lambda(lambda x: tf.slice(x,[0,0],[-1,N]),name='Get_Input')(shift_input)

    # Linear dim reduction
    PCA_input=Lambda(lambda x: tf.einsum("ij,jk->ik",x,Utens), name='PCA_input')(shift_input)

    # Nonlinear dim reduction
    encode=PCA_input
    for i in range(len(hiddenin)):
        encode=layers.Dense(hiddenin[i],activation=actin[i],name='Dense_In'+str(i+1))(encode)
    
    # Add linear and nonlinear dim reduction
    PCA_input_trunc=Lambda(lambda x: tf.slice(x,[0,0],[-1,trunc()]))(PCA_input)
    hidden=layers.Add(name='Hidden_Layer')([encode, PCA_input_trunc])

    # Save state for predicting phase evolution
    phase=hidden
    
    ########################################################################################
    # Timestepping
    ########################################################################################
    for i in range(len(timestep)):
        hidden=layers.Dense(timestep[i],activation=act_step[i],name='Timestep'+str(i+1))(hidden)

    ########################################################################################
    # Decoder
    ########################################################################################
    # Linear dim increase
    paddings=tf.constant([[0,0,],[0,N-trunc()]])
    PCA_output=Lambda(lambda x: tf.pad(x,paddings,'CONSTANT'), name='PCA_Int_Output')(hidden)

    # Nonlinear dim increase
    decode=hidden
    for i in range(len(hiddenout)):
        decode=layers.Dense(hiddenout[i],activation=actout[i],name='Dense_Out'+str(i+1))(decode)
    
    #Append tied layers on top (This was used for the additional loss in eq 4 during autoencoder training)
    conc=hidden
    for i in range(len(hiddenout)):
        conc=layers.Dense(hiddenout_PCA[i],activation=actout[i],name='Dense_Out_Reg'+str(i+1))(conc)

    decode=layers.Concatenate(name='Concatenate')([conc,decode])

    # Add PCA results to NN results
    PCA_output=layers.Add(name='PCA_Output')([decode, PCA_output])

    # Map back into the full space
    main_output=Lambda(lambda x: tf.einsum("ij,kj->ik",x,Utens), name='Output')(PCA_output)

    ########################################################################################
    # Phase shift
    ########################################################################################
    for i in range(len(hidden_phase)):
        phase=layers.Dense(hidden_phase[i],activation=act_phase[i],name='Dense_In_Phase'+str(i+1))(phase)
    
    phase=Lambda(lambda x: x*dtheta_std+dtheta_mean)(phase)
    phase_output=layers.Add()([phase,phase_input])

    # Shift back with phase    
    main_output=layers.Concatenate(name='Concatenate_Final')([main_output,phase_output])
    main_output=Lambda(Shift_Back,name='MoS_Out')(main_output)

    # Build model
    model=Model(inputs=main_input,outputs=main_output)
    
    return model,EPOCHS,optimizer
