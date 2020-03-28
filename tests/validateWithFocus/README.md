I want to make sure that with filamentary coils and an identical surface as FOCUS, FOCUSADD converges to the same (or very very similar due to the way I discretized the surface) solution of the coils. To that end, I perform an optimization of filamentary coils around the default axis which has a circular cross-section but non-circular axis shape. I need to save:
(i) the loss value of the optimization as a function of time
(ii) the Fourier coefficients on the coils
(iii) a picture of what the coils look like
(iv) a Poincare plot with the magnetic field created by the coils, which should be practically identical to the Poincare plot created by FOCUS. 

The inputs to the run are:

numCoils = 8
numTheta = 32
numZeta = 128
numSegments = 32
numFourierCoils = 5 (0th component plus 4 fourier components)
radiusCoils = 2.0 (the initial radius of the coils is twice the radius of the surface)
learningRate = 0.001
numIter = 500
weightLength = 0.1

The axis file is in focusadd/initFiles/axes/defaultAxis.txt. It has an epsilon value of 1.0, a minor radius of 0.1, N_rotate = 0 and zeta_off = 0.0. 

