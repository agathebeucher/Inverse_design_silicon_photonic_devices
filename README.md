# Inverse design of sSWG waveguides on silicon photonic devices

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## CONTENTS 
* [Structure](#STRUCTURE)
* [Project overview](#PROJECT-OVERVIEW)
* [Getting started](#getting-started-on-your-machine)
* [Models](#MODELS)
    * [EDA](#i--EDA)
    * [Feedforward network](#ii--feedforward-model-ffn)
    * [Genetic Algorithme](#iii--genetic-algorithm)

## STRUCTURE

### inverse_design_silicon_photonic_devices/
- **EDA/**
    - `load_data.py`
    - `normalize_data.py`
- **Feedforward_network/**
    - `feedforward_model_trained_gpu_5000.pth`
    - `feedforward_network_load.py`
    - `feedforward_network_train.py`
    - `feedforward_network_model.py`
-  **GA**
    - `approx_gauss.py`
    - `balayage_k.py`
    - `ga_evaluate.py`
    - `ga_model.py`
- **Inverse_design_network/tandem_network**
    - `Ìnverse_network_model.py`
    - `Inverse_network_train.py`
-  **results**
    - `figures/`
    - `result_param.txt`

## PROJECT OVERVIEW 

This project for nanophotonics design optimization involves developing a **feedforward neural network** and a **genetic algorithm** to optimize the design of nanophotonics silicon subwavelength grating waveguide. This tool aims to predict and tune the design parameters of a waveguides to reach a specific refractive index at a given wavelength.

## GETTING STARTED ON YOUR MACHINE

You can find the raw dataset from this [link](https://drive.google.com/file/d/1MrYbl_xirYWJZCTmyr7kOeqM50SCQTUO/view?usp=sharing), you can download it and place the dowloaded dataset in a "data" folder at the root folder and make sure the unzip file is named "NN_training_combine_new.csv", but you won't need it to run the rest : 

1. **Download the repository** : Clone or download this Git repository to your local machine
2. **Install python** : Ensure that you have Python and the pip command installed on your system.
3. **Install dependencies** : Run the script *requirements.txt* to install all necessaries dependencies by executing in your terminal the following command : 
```
pip install -r requirements.txt
```
Once you have everything set up, you can run the project by executing the following command in your terminal with the desired values of **effective index** at a given **wavelength** in *nanometers* :
```
python3 main.py --n_desired {n_value} --wavelength_desired {wavelength_value}
```
If you want, you can also fix the value of the width (in nm) and/or the pitch (in nm), but those arguments are optionnal.

For instance, you can run in your terminal: 
```
python3 main.py --n_desired 1.5 --wavelength_desired 1550 --fixed_w 430.0 -- fixed_pitch 300.0
```

Make sure your values are *floats*.

## MODELS

This project includes a feedforward neural network model that predicts the frequency spectrum of the electric field for a nanophotonic structure. The model takes four design parameters as inputs:

- **w** : Width of the waveguide
- **DC** : Duty cycle
- **Pitch** : Distance between adjacent elements
- **k** : Wave vector (deduced from the value of n_desired and f_desired)

<p align="center"><img src="images/1-SWGwaveguide.png" height="200"><p>
<p align="center"><I>Silicon-on-insulator waveguide with a sub-wavelength grating etched longitudinally or transversely</I></p>
The model is based on the idea that, using these four parameters, the network predicts 5000 values of the electric field spectrum, from which the resonance frequency and the effective refractive index of the structure can be derived.

However, the four parameters and the effective index are not directly linked. Using FDTD simulation, we predict the frequency spectrum of the waveguide based on these design parameters, and then obtain the effective index by extracting the resonance frequency and k. 

### I- EDA
Our dataset is made of the frequency spectrums for various combinations of design parameters, results of various FDTD simulations : 
In other words : 
- X_data=[w,DC,pitch,k] -> 4 values corresponding to the four designs parameters 
- y_data=[..,..,..] -> 5000 values of the electrical field for frequency values 

First, we filter our data to keep only the frequency spectrums that shows one peak (|E|>0.01). We then normalize both X_data and y_data (see `EDA/normalize_data`).

### II- Feedforward model (FFN)

Due to the one-to-many nature of the problem, we cannot directly predict the four parameters from one effective index since multiple designs can correspond to a single effective index.

We start by predicting the frequency spectrum corresponding to four design parameters. This is done using a feedforward network as our response prediction network. Its architecture is defined in `Feedforward_network/feedforward_network_model.py`. This fully connected network has six layers, with hyperparameters like learning rate and hidden sizes optimized using Optuna. 

<p align="center"><img src="images/2-FFN.jpg" height="400"><p>
<p align="center"><I>Feedforward Neural Network (FFN) architecture with four design characteristic parameters of the SWG as input and 5000 values of the electric field for multiple frequency values as output</I></p>

The trained model, that is to say the state of the weights and biases after training, is saved at `Feedforward_network/feedforward_model_trained_gpu_5000.pth`. You can load it thanks to `feedforward_network_load()` defined in `Feedforward_network/feedforward_network_load.py`.

This model is able the predict the correct the effective index (error<0.01) in 95% of the time (evaluated on a test dataset of size 390). In this figure, in blue is represented the spectrum simulated by FDTD and in orange the spectrum predicted by the FFN model : 

<p align="center"><img src="images/4-Train_Val_loss_over_Epochs_Response_network.png" height="300"><p>
<p align="center"><img src="images/3-Result_testdataset_FFN_5000_frequencies.png" height="600"><p>
<p align="center"><I>Comparison of predicted spectra (orange) and actual spectra (blue) for examples from the test dataset for the FFN</I></p>

### III- Genetic algorithm (GA)
Now that we have a model that can predict the frequency spectrum based on four input design parameters, we use a genetic algorithm to generate individuals better suited to reach the values that we want : n_desired and f_desired. Here are the main steps of the process : 

- Input a desired refractive index n and a specific frequency f.
- Generate combinations of design parameters (w, DC, pitch) using a genetic algorithm within a range.
- For each combination, sweep through multiple values of k.
- Use the feedforward model to obtain the resonance frequency and the corresponding n for each k.
- Plot n as a function *F(f_res)* of the resonance frequency f.
- Evaluate the n obtained with the desired_freqquency for this function *F(f_res)*
- Calculate the error between the n_desired and n_obtained
- Select the individuals minimizing the difference between the obtained n from the curve and the desired n
- Return the optimal values for w, DC, and pitch.

### IV- Performance
This model has been evaluated on a test dataset of 25 data, that have been generated by the GA+FFN, and then compared to the results of an FDTD simulation. In 19 cases out of 25, the model was able to predict values that generated an effectiv index whose precision <0.01. 

In other words, this method is efficient **76% of the time**.

# Contact

Pour tout question, n'hésitez pas à envoyer un message !

> Agathe **BEUCHER**
>
> Agathe.beucher@ensta-paris.fr
>
> +33 6 70 33 88 06