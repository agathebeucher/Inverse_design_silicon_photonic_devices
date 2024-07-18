from scipy.optimize import curve_fit
import numpy as np
import time

def approx_gauss(y_data):
    '''
    This function approximates a Gaussian function to the given data and computes the mean squared error (MSE) of the fit.
    '''
    # Generate a list of 5000 values ranging from 0 to 4999 using np.linspace
    x_data = np.linspace(0, 5000, 5000, dtype=int)

    # Define the non-centered Gaussian function
    def gaussienne(x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    try:
        # Fit the Gaussian function to the data
        popt, pcov = curve_fit(gaussienne, x_data, y_data, p0=[np.max(y_data), np.argmax(y_data), np.std(y_data)])
        
        # popt contains the optimal values for a, mu, and sigma
        a_opt, mu_opt, sigma_opt = popt

        # Compare the differences between the fitted curve and the real data
        y_fit = gaussienne(x_data, *popt)
        residuals = y_data - y_fit

        # Compute the mean squared error (MSE)
        mse = np.mean(residuals ** 2)
        return mse, y_fit
    
    except RuntimeError as e:
        # Return None if the fitting fails
        return None, None
