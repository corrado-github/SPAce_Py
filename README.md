# SPAce_Py
Python version of SP_Ace with use of Neural Networks for the GCOG library

SP_Ace (Stellar PArameters and Chemical abundances Estimator, Boeche & Grebel, 2016, A&A 587, 2) is a software for automated analysis of stellar spectra.
It derives stellar parameters (such as effective temperature, gravity, and metallicity) and chemical abundances for FGK stars. SP_Ace constructs the spectrum model
retrieving information of the central wavelengths and strengths of the spectral lines from the General-Curve-Of-Growth (GCOG) library specifically
prepared and, by assuming a line profile, produces a spectrum model from this information. It produces many spectrum models with
different stellar paramenets and elemental abundances of several elemens looking for the best match with the observed spectrum using a chi^2 minimization routine.
The original SP_Ace FORTRAN95 code can be found at http://dc.g-vo.org/SP\_ACE.

The present version of SP_Ace is written in Python. Unlike the original FORTAN code, this version uses Neural Network (NN) models for 
the GCOG of the absorption lines of the line list. This solution can make the code easier because every GCOG NN model cover the full parameter space.

The present code is work in progress.
