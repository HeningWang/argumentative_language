# Analysis

This folder contains all analysis code, model fitting, and visualization scripts for the argumentative language research project.

## Overview

This analysis implements and evaluates computational models of argumentative language from both speaker and listener perspectives. The models use Bayesian inference (PyMC3) to understand how speakers select arguments and how listeners interpret them.

## Main Analysis Notebooks

### Experimental Data Analysis
- **`analysis_pilot_exp1.ipynb`** - Analysis of pilot study and Experiment 1 data with fixed array sizes
- **`analysis_exp2.ipynb`** - Analysis of Experiment 2 data with variable array sizes
- **`analysis_listenerside.ipynb`** - Analysis of listener-side experimental data
- **`excluded_models.ipynb`** - Alternative models considered during development (tested on pilot data)

### Model Inference and Computation
- **`run_inference_speaker_models.ipynb`** - Runs Bayesian inference for speaker models
- **`argstrengths.ipynb`** - Computes and validates argument strength measures on full state space
- **`listenerside_item_generation.ipynb`** - Generates and evaluates items for listener-side experiments
- **`tests_and_designs.ipynb`** - Model validation, posterior predictive checks, and experimental design testing

## Supporting Code

### `functions/` Directory
- `argstrengths.py` - Argument strength calculations
- `argstrengths_fullstatespace.py` - Full state space argument strength computations
- `data_functions.py` - Data loading and cleaning utilities
- `helper_functions.py` - General utility functions (normalization, traces, etc.)
- `models_fixedarray.py` - Models for fixed array size experiments
- `models_variablearray.py` - Models for variable array size experiments
- `models_listener.py` - Listener-side models
- `plotting.py` - Visualization and plotting functions

## Output Directories

- **`argstrengths/`** - Pre-computed argument strength values (including full state space calculations)
- **`figs/`** - Generated figures and visualizations
- **`analysis_exp_listener_side/`** - R analysis code for listener-side experiment
- **`item_generation_listener_side/`** - Materials and data for listener-side item generation

## Environment Setup

The analysis environment is defined in `environment.yml`. Key dependencies include:

- Python 3.8
- PyMC3 3.11.4 (Bayesian inference)
- Theano 1.1.2 (computational backend)
- Arviz 0.12.1+ (visualization and diagnostics)
- pandas, numpy, matplotlib, seaborn (data manipulation and plotting)


## Data Paths

Analysis notebooks reference data from:
- `../data/data_pilot/` - Pilot study data
- `../data/data_experiment1/` - Experiment 1 data
- `../data/data_experiment2/` - Experiment 2 data
- `../data/data_listenerside/` - Listener-side experiment data
- Model traces stored externally (not in git due to file size >100MB)

## Workflow

1. Data cleaning and preparation (via `data_functions.py`)
2. Model specification and inference (PyMC3 models in `run_inference_speaker_models.ipynb`)
3. Argument strength computation (`argstrengths.ipynb`)
4. Model comparison and validation (`tests_and_designs.ipynb`)
5. Experimental analysis (`analysis_exp2.ipynb`, `analysis_listenerside.ipynb`)
6. Figure generation (saved to `figs/`)

## Notes

- Trace files from model fitting are large (>100MB) and stored separately
- Some argument strength calculations on full state space are computationally intensive and pre-computed
- The codebase supports both fixed and variable array size experimental designs