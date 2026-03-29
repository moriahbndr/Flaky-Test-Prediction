# COSC 490 Research Project
#
### PROJECT OVERVIEW ####
This project focuses on flaky test predictioins without reruns using:
* static features
* lightweight dynamic features

Dataset used for training are both FlakeFlagger and iDFlakies
---------------------------------------------------------------------------------------

# Instructions for getting started after cloning the repo

## Set Up ##

1. Create a virtual environment and install packages ()

    MAC or Linux
    - (python3 or python) -m venv venv
    - source venv/bin/activate

    Windows
    - python -m venv venv
    - venv\Scripts\activate
    - pip install -r packages.txt

* VSCode terminal on MAC example :  
    python3 -m venv
    source venv/bin/activate
    pip install -r packages.txt

# ---------------------------------------------------------------------------------------

#### PROJECT STRUCTURE AS FOLLOWS: ####

data/
    input_data/ (FlakeFlagger and iDFlakies dataset uploads here in this folder)
    processed/  (output of the feature extraction pipeline here.... (more specifically, src/features/build.py output))

results/
    models/ (trained XGBoost model - updates after every src/model.training.py run)

src/
    features/ (creation of static and dynamic features here)

    data_check.py - not necessary to overall project, but can be used to check data outputs seperately, column names, ect.
    model_training.py - building and training the model

packages.txt - contains all of the packages to be installed during set up of this project    
---------------------------------------------------------------------------------------

#### RUNNING THE PROJECT ####

1. RUN src/features/build.py (loads the dataset + creates the static & dynamic features + saves the processed dataset)
2. RUN src/model_training.py (training the model XGBoost and stores the results in restuls/models/xgboost_model.pkl)

* near completiong of project, will only need to create and run main for processes to run in order

# ---------------------------------------------------------------------------------------

## Data Set table columns ##

FlakeFlagger
- Project
- Test
- IsFlaky
- NumFailingRuns
- NumPassingRuns
- FirstFailingRunID
- FirstPassingRunID
- UniqueFailingExceptionTypes

* these can be found on line 1 in the FlakeFlagger dataset located in data/input_data/flakeflagger_results.csv

iDFlakies
(dataset is added in project in data/input_data/idFlkies_dataset.csv, dataset columns to be added into README.md soon)
---------------------------------------------------------------------------------------

#### STATIC FEATURES ####
- FunctionNameLength
- ClassNameLength
- PackageLength
- SleepOrWaitInFunction
- AsyncInFunction
- TimeOrRandomInFunction


#### LIGHTWEIGHT DYNAMIC FEATURES ####
- NumFailingRuns
- NumPassingRuns
- FirstFailingRunID
- FirstPassingRunID
- UniqueFailingExceptionTypes
- TotalRuns
- FailRatios
- PassRatioi
- AnyFailures
- AnyPassingRuns

# ---------------------------------------------------------------------------------------

#### MODEL OUTPUTS ####
- Precision
- Recall
- F1 Score
- Confusion Matrix

results are saved in results folder

# ---------------------------------------------------------------------------------------
