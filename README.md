# Inverse design of silicon photonic devices

## GOAL 

The goal of the project is to predict four design parameters of a nanosilicon waveguide based on the value of the effective index desired. 
In order to run it, all you have to do is.

## GETTING STARTED

- Download this git
- Download the data : https://drive.google.com/file/d/1MrYbl_xirYWJZCTmyr7kOeqM50SCQTUO/view?usp=sharing
- Transfer the data downloaded in the folder "data" under the name "NN_training_combine_new.csv"
- You will find all of the dependency you will need installed in order to run this project : *requirements.txt*, run this script in a terminal

Once you have installed and downloaded everything, all you have to do is run in your terminal, with the desired values :
```
python3 main.py {n_value}
```
n_value must be a list of float, for instance : n_value=*[w_value, DC_value, pitch_value, k_value]*

## MODELS
We are here using inverse design in order to achive this prediction : 

### I- Feedforward model

### II- Inverse Design 
####    a- Tandem Network
####    b- Genetic algorithm


