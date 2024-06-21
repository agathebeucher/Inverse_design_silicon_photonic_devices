import zipfile
import os

def archive_csv(csv_file_path, zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file_path, arcname=os.path.basename(csv_file_path))

# Chemin du fichier CSV à archiver
csv_file_path = '/home/beucher/Documents/PRE/PRE/data/NN_training_combine_new.csv'

# Chemin du fichier ZIP à créer
zip_file_path = '/home/beucher/Documents/PRE/PRE/data/NN_training_combine_new.zip'

archive_csv(csv_file_path, zip_file_path)

print(f"Fichier {csv_file_path} archivé dans {zip_file_path}")
