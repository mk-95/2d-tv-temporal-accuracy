# How to Run the Code

the files called error_fun_<integrator>.py contain the implementation of the <integrator> scheme. these files return the value
of the x velocity at the end in order to compute the temporal order of accuracy. 

you run these files from richardson_extrapolation.py, at the end of the run the global error is shown on a plot

the file functions.py contain the functions used in each of the error_fun_<integrator>.py the guessses are implemented 
in the function `guess`. 

The file singleton_classes.py contain the information about the grid and the different types of time integrator coefficients 
