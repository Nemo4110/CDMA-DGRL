- Get the dataset from the [official website](https://mimic.mit.edu/) of mimic-iii dataset:
  - the completion of an ethics course is required.
  - then you need to sign the *PhysioNet Credentialed Health
Data Use Agreement 1.5.0 - Data Use Agreement for the
MIMIC-III Clinical Database (v1.4)* to get the accessment of dataset. 
- data preprocessing scripts are located in the dataset/preprocess*
- The scripts for running the experiments on Google Colab are located in the .ipynb files under the notebook directory:
  - run_backbone.ipynb
  - run_baseline*.ipynb
- Paths should be modified in utils/constant.py (we used Google Drive and mounted it to the Colab runtime).