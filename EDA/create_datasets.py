import numpy as np

y_data_array_5000_normalized = normalize_with_max_peak(y_data_array_5000)

# Diviser les données en ensembles de formation, de validation et de test
X_train_5000_normalized, test_val_X_5000_normalized, y_train_5000_normalized, test_val_Y_5000_normalized = train_test_split(X_data_array_5000_normalized, y_data_array_5000_normalized, test_size=0.2, random_state=42)
X_test_5000_normalized, X_val_5000_normalized, y_test_5000_normalized, y_val_5000_normalized= train_test_split(test_val_X_5000_normalized, test_val_Y_5000_normalized, test_size=0.5, random_state=42)
