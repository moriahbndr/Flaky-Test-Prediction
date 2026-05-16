# COSC 490 Research Project
#
### PROJECT OVERVIEW ####
This project focuses on flaky test predictions without reruns using:
* static features
* lightweight dynamic features

Datasets used for training are both FlakeFlagger and iDFlakies
---------------------------------------------------------------------------------------

> NOTE: the first time you run main.py it will take a few minutes (sometimes 10 - 15 min + ) — build_smells.py has to fetch
> individual Java test files from GitHub for the entire FlakeFlagger dataset. after that first run
> everything gets cached locally so re-runs are instant

---------------------------------------------------------------------------------------

# Instructions for getting started after cloning the repo

## Set Up ##

1. Create a virtual environment and install packages

    MAC or Linux
    - (python3 or python) -m venv venv
    - source venv/bin/activate
    - pip install -r packages.txt

    Windows
    - python -m venv venv
    - venv\Scripts\activate
    - pip install -r packages.txt

* VSCode terminal on MAC example :  
    python3 -m venv venv
    source venv/bin/activate
    pip install -r packages.txt

# ---------------------------------------------------------------------------------------

#### PROJECT STRUCTURE AS FOLLOWS: ####

data/
    input_data/  (FlakeFlagger and iDFlakies datasets found here)
    processed/   (output of the feature extraction pipeline — CSVs saved here after running)

results/
    models/   (trained XGBoost model — updates every time model_training.py runs)
    figures/  (calibration curves and feature importance plots per experiment)
    tables/   (model_metrics.csv and cross_project_metrics.csv saved here)

src/
    features/
        static_features.py     (shared keyword/feature logic used by both build scripts)
        build_flakeflagger.py  (extracts static + lightweight dynamic features from FlakeFlagger)
        build_smells.py        (fetches Java source from GitHub and extracts test smell features)
        build_idflakies.py     (extracts static features from iDFlakies for cross-project eval)

    data_check.py      (can be used for checking data outputs but project nevver calls this file for use)
    model_training.py  (trains all experiments and runs cross-project evaluation on iDFlakies)

main.py       (runs the full pipeline in order, just run this)
packages.txt  (lists all packages needed, install these during setup)

# ---------------------------------------------------------------------------------------

#### RUNNING THE PROJECT ####

run main.py from the project root and it handles everything in order:

    python main.py  (or python3 main.py on mac)

it runs these steps automatically:
1. src/features/build_flakeflagger.py  — builds features from the FlakeFlagger dataset
2. src/features/build_smells.py        — fetches Java source files and extracts smell features (takes a while the first time, cached after)
3. src/features/build_idflakies.py     — builds static features from the iDFlakies dataset
4. src/model_training.py               — trains all experiments and evaluates on both datasets

if a step fails it will stop and print out which one

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
- Project URL
- SHA Detected
- Module Path
- Fully-Qualified Test Name (packageName.ClassName.methodName)
- Category  (type of flakiness — ID, OD, NOD, TD, etc.)
- Status
- PR Link
- Notes

* located in data/input_data/idFlakies_dataset.csv
* all entries in iDFlakies are confirmed flaky tests (IsFlaky = 1)
* used for cross-project evaluation only — the model is never trained on this dataset

---------------------------------------------------------------------------------------

#### STATIC FEATURES ####
- FunctionNameLength
- ClassNameLength
- PackageLength
- FunctionWordCount
- FunctionHasDigits
- SleepOrWaitInFunction
- AsyncInFunction
- TimeOrRandomInFunction
- NetworkInFunction
- FileIOInFunction
- DatabaseInFunction
- UIBrowserInFunction
- RetryFlakeInFunction
- SleepOrWaitInClass
- AsyncInClass
- TimeOrRandomInClass
- NetworkInClass
- FileIOInClass
- DatabaseInClass
- UIBrowserInClass
- NetworkInPackage
- FileIOInPackage
- DatabaseInPackage
- UIBrowserInPackage


#### LIGHTWEIGHT DYNAMIC FEATURES ####
- NumFailingRuns
- NumPassingRuns
- FirstFailingRunID
- FirstPassingRunID
- UniqueFailingExceptionTypes
- TotalRuns
- FailRatio
- PassRatio
- AnyFailures
- AnyPassingRuns
- BothPassAndFail
- AlwaysFails
- FailOnFirstRun
- EarlyFailure
- ExceptionDiversityRatio

#### TEST SMELL FEATURES (from Java source) ####
- assertion_roulette
- conditional_test_logic
- eager_test
- fire_and_forget
- indirect_testing
- mystery_guest
- resource_optimism
- test_run_war
- num_asserts
- test_length

# ---------------------------------------------------------------------------------------

#### EXPERIMENTS ####
- flakeflagger_raw          — uses raw run history columns (leaky, upper bound baseline)
- smell_only                — test smell features only
- static_v2                 — name-based static features only, no run history
- static_v2_plus_dynamic    — static features + derived run stats
- flakeflagger_static       — smell features + churn (FlakeFlagger paper's approach)

#### MODEL OUTPUTS ####
- Precision
- Recall
- F1 Score
- Brier Score
- Confusion Matrix
- Misclassification Cost
- Cross-project Recall on iDFlakies (static experiments only)

results are saved in the results/ folder

# ---------------------------------------------------------------------------------------
