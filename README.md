# least_squares
Benchmark different algorithms for least squares and instrumental variable estimators.

Introduction
------------

The goal of this project is to benchmark different ols and iv implementations. There are two aspects to the project, one is to see how different implementations perform when 
the number of observations, variables and instruments increase and the other aspect is to see how accurate the estimator of different implementations become when there exists
data issues like collinearity and weak instruments and efficiency issues like how many instruments to include to effectively tackle an endogeneity in the data (just identified 
or over-identified iv). Perfomance is defined as the time it takes for the implementation(s) to be executed.


Challenges & Goals
------------------

The goal of this project is to demonstrate how different observation, variable and instrument sizes can affect the the perfomance of an ols/iv implementation as well as the accuracy
of the estimator in relation o the true beta. The main challenges faced during this project is the difficulty in constructing valid covariance matrices for the cases of instrumental 
variables - especially to demonstrate the weak instrument case. As discovered in the proccess of completing the project, it is challenging to construct positive definite matrices with 
simulated data that will have all the properties required to check the accuracy of the implementations under weak instruments. Further improvement is required in this aspect so to make it
possible to generate valid covariance matrices for the case of weak instruments.


The ideal product of this project would:
* contain perfomance plots for ols that display an increase in runtime as observations and variables increase.
* contain perfonance plots for iv that display an increase in runtime as observations, variables, and number of instruments increases.
* contain accuracy plots for ols that show that as colliniarity increases, the accuracy of the estimator in relation to the true beta decreases using the root mean squared error approach.
* contain accuracy plots for ols that show that as colliniarity and number of instruments increases, the accuracy of the estimator in relation to the true beta decreases using the root 
	mean squared error approach.
* contain accuracy plots for iv showing that as the strength of the instruments decreases, the accuarcy of the estimator should also decrease (**to revert back to in the future).


Structure of the project
-----------------------------------

There are four main steps involved in the project:
1. Ols and iv implementations
2. Generating the data
3. Timing and perfomance plots
4. Accuracy plots

In the first step, l write functions for all the ols and iv implementations in two seperate modules. This follows from the standard textbook ols and iv formulae. After this l implement 
a data generating function (in a seperate module) to generate the different datasets required for the project. To generate the perfomance plots, l implement timing functions to get runtimes
for each implementation over different observations, variables and instruments and lastly, l implement the root mean squared error method to check the accuracy of all ols and iv 
implementations.


Structure of the project
-------------------------------
The project is organised into a code folder that contains all the modules you need to generate the benchmark plots. All plots will be generated and saved in a seperate bld folder that
will be created as a result of invoking code embeded in the benchmark modules. 

To reproduce the results mentioned above you will need to:
1. Clone/download my repository to your local pc/machine
2. run the benchmark_ols.py and benchmark_iv.py modules contained in the code folder via anaconda using the comand 'python benchmark_ols.py' and 'python benchmark_iv.py'.
3. Check for the generated .png files in the bld folder.

These modules depend on the following modules (also contained in the code folder)
* timing.py
* generate_data.py
* iv.py
* ols.py


Requirements
------------
This project as alluded to earlier requires:
* anaconda or any terminal from which you can execute python scripts
* your preffered python version installed

These are all the requirements to run the code. 
