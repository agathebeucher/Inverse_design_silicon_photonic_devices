# Inverse design of silicon photonic devices

## GOAL 

The goal of the project is to predict four design parameters of a nanosilicon waveguide based on the value of the effective index desired. 
In order to run it, all you have to do is.

## GETTING STARTED ON YOUR MACHINE

- Download this git
- Download the dataset here : https://drive.google.com/file/d/1MrYbl_xirYWJZCTmyr7kOeqM50SCQTUO/view?usp=sharing
- Transfer the data downloaded in the folder "data" under the name "NN_training_combine_new.csv"
- You will find all of the dependency you will need installed in order to run this project : *requirements.txt*, run this script in a terminal

Once you have installed and downloaded everything, all you have to do is run in your terminal, with the desired values :
```
python3 main.py {n_value}
```
n_value must be a list of float, for instance : n_value=*[w_value, DC_value, pitch_value, k_value]*

## MODELS
We are here using inverse design in order to predict four design parameters of a nanophotonic silicon waveguide based on the value of the effective index desired.
The four parameters and the effective index are not directly linked, thanks to a FDTD simulation, we predict the frequency spectrum of the waveguide based on those design parameters, and then, we obtain the effective index by extracting the resonance frequency and the k. 
In our dataset, we have for a combination of design parameters the corresponding frequency spectrum. 
In other words : 
- X_data=[w,DC,pitch,k] -> 4 values corresponding to the four designs parameters 
- y_data=[..,..,..] -> 5000 values of the electrical field for frequency values beteween ... and ...

### I- EDA
We first filter our data to keep the frequency spectrums that show a peak (>0.01). We than normalize X_data and y_data. 

### II- Feedforward model

Because of the nature of the problem (one-to-many), we cannot predict directly the four parameters from one effective index, because for one effective index we have various possible design. We have to use an inverse design. 

We start by predicting the frequency sprectrum corresponding to four design parameters : this is the response prediction network that we will call the feed-forward network, whose architecture can be found here : *Feedforward_network/feedforward_network_model.py*

We use a fully connected network with four layers, and whose hyperparameters learning_rate, hidden_sizes have been optimized thanks to optuna.
The model has already been trained and saved here : *Feedforward_network/feedforward_network_trained.pth*

### III- Inverse Design 
Now that we are able to predict the frequency response to four design parameters, we want to do the inverse mechanism. There are two possible options :

####    a) Tandem Network
First, we use another fullyconnected network that we optimise and train through a tandem network, wich means that we use the feedforward network at the output of the inverse design network in order to transform our problem to a one-to-one problem, meaning that we know teach our model to learn how to fit to a frequency response, rather than the fous parameters, whose response is not unique : 

####    b) Genetic algorithm
We can also use a genetic algorithm, that will generate various combination of parameters and wich will test all of them using the feed-forward newtork in order to return the one better fit to give the desired frequency response.


