Here, I test whether the coils created by the Frenet frame rotate too quickly when the number of Fourier modes in the coils is relatively high. I will compare the Frenet frame to the COM frame.

The inputs to the run are:

numCoils = 16

numTheta = 32

numZeta = 128

numSegments = 64

numFourierCoils = 10 (0th component plus 9 fourier components)

numFourierRotate = 0

numNormalRotate = 3

numBinormalRotate = 3

radiusCoils = 2.0 (the initial radius of the coils is twice the radius of the surface)

learningRate = 0.0005

numIter = 25

weightLength = 0.1

optimizer = momentum



The axis file is in focusadd/initFiles/axes/ellipticalAxis4Rotate.txt. It has an epsilon value of 0.5, a minor radius of 0.1, N_rotate = 4 and zeta_off = 0.0.

