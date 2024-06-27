# Inverse design of silicon photonic devices

## STRUCTURE

### inverse_design_silicon_photnic_devices/
- **EDA/**
    - `filter_data.py`
    - `normalize_data.py`
- **Feedforward_network/**
    - `feedforward_model_trained.pth`
    - `feedfroward_network_evaluate.py`
    - `feedforward_network_train.py`
    - `feedforward_network_model.py`
- **Inverse_deisgn_network/**
    - **GA/**
        - `ga_evaluate.py`
        - `ga_model.py`
    - **Tandem_network/**
        - `Ìnverse_network_evaluate.py`
        - `Ìnverse_network_model.py`
        - `Inverse_network_train.py`

## GOAL 

The goal of the project is to predict four design parameters of a nanosilicon waveguide based on a desired effective index value. 
To get started, follow the instructions below :*
## GETTING STARTED ON YOUR MACHINE

You can find the raw dataset from this [link](https://drive.google.com/file/d/1MrYbl_xirYWJZCTmyr7kOeqM50SCQTUO/view?usp=sharing), you can download it and place the dowloaded dataset in a "data" folder at the root folder and make sure the unzip file is named "NN_training_combine_new.csv", but you won't need it to run the rest : 

1. **Download the repository** : Clone or download this Git repository
2. **Install librairies** : Run the script *requirements.txt* to install all necessaries dependencies : 
```
pip install -r requirements.txt
```

Once you have everything set up, you can run the project by executing the following command in your terminal with the desired values of **effective index** at a given **wavelength** in *nanometers* :
```
python3 main.py --n_desired {n_value} --wavelength_desired {wavelength_value}
```
Make sure your values are *floats*.

## MODELS
This project uses inverse design to predict four design parameters of a nanophotonic silicon waveguide based on a desired effective index value. The four parameters and the effective index are not directly linked. Using FDTD simulation, we predict the frequency spectrum of the waveguide based on these design parameters, and then obtain the effective index by extracting the resonance frequency and k. 
Our dataset contains the frequency spectrum for various combinations of design parameters, results of various FDTD simulations : 
In other words : 
- X_data=[w,DC,pitch,k] -> 4 values corresponding to the four designs parameters 
- y_data=[..,..,..] -> 5000 values of the electrical field for frequency values beteween ... and ...

### I- EDA
First, we filter our data to keep only the frequency spectrums that show a peak (>0.01). We then normalize both X_data and y_data.

### II- Feedforward model

Due to the one-to-many nature of the problem, we cannot directly predict the four parameters from one effective index since multiple designs can correspond to a single effective index. Instead, we use an inverse design approach.

We start by predicting the frequency spectrum corresponding to four design parameters. This is done using the response prediction network, known as the feedforward network. Its architecture is defined in *Feedforward_network/feedforward_network_model.py*. This fully connected network has four layers, with hyperparameters like learning rate and hidden sizes optimized using Optuna. 

The trained model, that is to say the state of the weights and biases after training, is saved at *Feedforward_network/feedforward_network_trained.pth*.

### III- Inverse Design 
Now that we can predict the frequency response to four design parameters, we want to reverse this process. There are two possible approaches:

####    a) Tandem Network
We use another fully connected network optimized and trained through a tandem network. This involves using the feedforward network at the output of the inverse design network, transforming our problem into a one-to-one problem. The model learns to fit a frequency response, rather than the four parameters, whose response is not unique.

####    b) Genetic algorithm
Alternatively, we can use a genetic algorithm to generate various combinations of parameters. This algorithm tests all combinations using the feedforward network to determine the best fit for the desired frequency response.

