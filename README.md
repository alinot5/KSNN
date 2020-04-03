# KSNN
Code used for evolving solutions of the Kuramoto-Sivashinsky equation with neural networks described in https://arxiv.org/abs/2001.04263.

To run clone this directory and run KSNN_Example.py. Vary time_units and u0 to run different initial conditions.

Notes:

This was built using tensorflow 1.14.0. Install this version if functions have depreciated.

If this error appears "OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized." or you're running in spyder and it appears to crash for no reason, then run "%env KMP_DUPLICATE_LIB_OK=TRUE" in the console.
