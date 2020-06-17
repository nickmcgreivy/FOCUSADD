In this directory, we compare the coils which are made from three different optimization methods. First, we look at filamentary coils. Second, we look at finite-build coils which have the COM frame and do not optimize their rotation. Third, we look at finite-build coils which have the COM frame and do optimize their rotation. 

The axis used is ellipticalAxis4Rotate, with 4-fold symmetry. The symmetry is not taken advantage of during the optimization.

The parameters used are:

NC = 20 (5 coils per section)
NS = 128
NFC = 6
NFR = 0 or 4
NZ = 128
NT = 32
length_weight = 0.1
learning_rate= 0.0001
optimizer = momentum
frame = com
num_iter = 200
nnr = nbr = 1 or 3
ln = lb = 0.015
nr = 0
momentum_mass = 0.9
output_file = filament, fixed_finite_build, or rotate_finite_build



Test #1: Filamentary

filament.hdf5
filament.txt




Test #2: Finite Build, No Rotate

fixed_finite_build.hdf5
fixed_finite_build.txt



Test #3: Finite Build, With Rotation

rotate_finite_build.hdf5
rotate_finite_build.txt




What is compared?

(i) The final loss values of the coils
(ii) The shapes of the coil centroids - are they very different? Does the final position depend on which method used?
(iii) The rotation profile of the finite-build coils - generate plots of coils, plus zoomed-in plots of a few coils at a time with a comparison.
(iv) The physics loss versus the weight loss for each - does one do better than another on one metric but not the other? Does a lower loss lead to lower physics loss?
