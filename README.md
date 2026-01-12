# Modeling Language Use with Argumentative Goals

This repository contains the code, data, and analysis for a research project investigating the production of quantifiers in argumentative contexts. The project mainly consists of experimental results and Bayesian Data Analysis with a variety of different models.

## Project Structure

- **data/**  
  Contains datasets from three experiments:
  1. A pilot study for model exploration.
  2. Experiment 1: fixed matrix size (5 students × 12 questions).
  3. Experiment 2: variable matrix sizes.
  
  Also includes `data-exploration.r` for data wrangling and plotting.

- **analysis/**
  Jupyter notebooks and supporting code for modeling and analyzing the experimental data:
  - `analysis_pilot_exp1.ipynb`: Analysis for pilot and Experiment 1.
  - `analysis_exp2.ipynb`: Analysis for Experiment 2.
  - `argstrengths.ipybn`: Visualization and comparison of argstrengths, also vis a vis the data.
  - `tests_and_designs.ipybn`: Various tests, plots, etc., that didn't naturally fit anywhere else.
  - `excluded_models.ipynb`: Models that were considered but didn't make it to the preregistration. This was cleaned up fairly quickly from a previous version, so some things might be slightly broken.
  - `functions/`: Python functions used in the notebooks.
  - `environment.yml`: Conda environment for reproducibility.


- **experiments/**  
  Implementation of the main experiment using [magpie](https://magpie-experiments.org/), a framework for running web-based behavioral experiments.  
  - To run locally:
    1. Install Node.js (v14.x, 16.x, 18.x, or 20.x) and npm (≥7.0.0).
    2. Run `npm install` in this directory.
    3. Use `npm run serve` to start the local server.

- **paper/**  
  The research paper describing the experiments and modeling.

## Installation & Requirements

- **Analysis:**  
  Use the provided `analysis/environment.yml` to create a conda environment:
  ```bash
  conda env create -f analysis/environment.yml
  conda activate argumentative_language
  ```
- **Experiment:**  
  See `experiments/speaker_side/README.md` for details. Requires Node.js and npm.

## Authors

- Fausto Carcassi
- Hening Wang
- Chris Cummins
- Michael Franke

## License

MIT License

Copyright (c) 2024 Fausto Carcassi, Hening Wang, Chris Cummins, Michael Franke

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 
