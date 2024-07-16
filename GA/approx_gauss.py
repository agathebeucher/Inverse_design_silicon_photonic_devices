from scipy.optimize import curve_fit
import numpy as np
import time

def approx_gauss(y_data):
    # Générer une liste de 5000 valeurs allant de 0 à 4999 avec np.linspace
    x_data = np.linspace(0, 5000, 5000, dtype=int)
    
    # Définir la fonction gaussienne non centrée
    def gaussienne(x, a, mu, sigma):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
    try :
        # Ajuster la fonction gaussienne aux données
        popt, pcov = curve_fit(gaussienne, x_data, y_data, p0=[np.max(y_data), np.argmax(y_data), np.std(y_data)])
        
        # popt contient les valeurs optimales pour a, mu et sigma
        a_opt, mu_opt, sigma_opt = popt

        # Comparer les écarts entre la courbe ajustée et les données réelles
        y_fit = gaussienne(x_data, *popt)
        residuals = y_data - y_fit

        # Calculer l'erreur quadratique moyenne (MSE)
        mse = np.mean(residuals**2)
        return(mse, y_fit)
    
    except RuntimeError as e:
        return None, None
