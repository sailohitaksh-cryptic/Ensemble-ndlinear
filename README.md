# NdLinear vs nn.Linear MNIST Benchmark

This repository contains a Jupyter Notebook (`tuned_assignment.ipynb`) that benchmarks a custom `NdLinear` layer against the standard PyTorch `nn.Linear` layer for image classification on the MNIST dataset. The primary goal is to evaluate the parameter efficiency and performance of `NdLinear` when leveraging its multi-dimensional input capabilities.

## Project Overview

The notebook compares two simple Multi-Layer Perceptron (MLP) models:

1.  **Baseline MLP:** Uses standard `nn.Flatten` followed by `nn.Linear` layers.
2.  **NdLinear MLP:** Uses the `NdLinear` layer to process the 2D spatial structure (`28x28`) of the input images directly in the first layer, followed by flattening and a final `nn.Linear` layer for classification.

The project includes:
* Data loading and preprocessing for MNIST.
* Model definitions for both baseline and `NdLinear` MLPs.
* Utility functions for training, evaluation, and parameter counting.
* Training and evaluation loop for the baseline model.
* **Hyperparameter tuning** for the `NdLinear` model (exploring different hidden dimensions, learning rates, and epochs).
* Training and evaluation of the **best** performing `NdLinear` configuration found during tuning.
* A final comparison and analysis of the results.

## Key Features

* Demonstrates the use of a custom layer (`NdLinear`) capable of handling multi-dimensional inputs directly.
* Provides a clear benchmark against a standard `nn.Linear` baseline.
* Includes hyperparameter tuning to optimize the custom layer's performance.
* Highlights the trade-offs between parameter count, model accuracy, and training time.

## Requirements

* Python >=3.12
* PyTorch
* Torchvision
* `ndlinear` library


## Usage

1.  Clone the repository:
    ```bash
    git clone https://github.com/sailohitaksh-cryptic/Ensemble-ndlinear.git
    ```
    ```bash
    cd Ensemble-ndlinear
    ```
    
2.  Install required packages:
    ```bash
    pip install torch torchvision torchaudio ndlinear
    ```
3.  Open and run the Jupyter Notebook:
    ```bash
    jupyter notebook benchmark_notebook.ipynb
    ```
    (Or open it in JupyterLab, VS Code, Google Colab, etc.)

    The notebook is structured sequentially. Running all cells will:
    * Set up the environment and data.
    * Train and evaluate the baseline model.
    * Perform hyperparameter tuning for the `NdLinear` model (this may take some time).
    * Train and evaluate the best `NdLinear` model identified.
    * Display the final comparison table and analysis.

## Results Summary

After hyperparameter tuning, the best performing `NdLinear` model was compared against the baseline:
```text
==================================================
       Final Benchmark Results Comparison
==================================================
Baseline Model (nn.Linear):
  Parameters:         101,770
  Test Accuracy:       0.9797 (97.97%)
  Training Time:       121.57s
--------------------------------------------------
Best NdLinear Model (Multidimensional Mode):
  Config:        Hidden=(18, 18), LR=0.001, Epochs=15
  Parameters:           4,294
  Test Accuracy:       0.9763 (97.63%)
  Training Time:       140.90s
--------------------------------------------------
Comparison (Best NdLinear vs Baseline):
  Parameter Reduction:      95.78%
  Accuracy Change:        -0.34 % points
  Training Time Ratio:       1.16x (Slower)
==================================================
```


## Analysis Highlights

* **Parameter Efficiency:** The `NdLinear` layer demonstrated significant parameter efficiency, reducing the model parameters by **~95.8%** compared to the baseline while processing the input's spatial structure.
* **Performance:** The optimized `NdLinear` model achieved accuracy (**97.63%**) highly comparable to the baseline (**97.97%**), with only a minor decrease of 0.34 percentage points.
* **Trade-off:** `NdLinear` provides an excellent trade-off, drastically reducing model size with minimal impact on predictive performance for this task, albeit with a slight increase in training time (~1.16x slower).
* **Conclusion:** `NdLinear` is confirmed as a compelling alternative for efficient deep learning on structured, multi-dimensional data, particularly when model footprint is a concern.

## Files

* `benchmark_notebook.ipynb`: The main Jupyter Notebook containing all code, experiments, and analysis.
* `README.md`: This file.

