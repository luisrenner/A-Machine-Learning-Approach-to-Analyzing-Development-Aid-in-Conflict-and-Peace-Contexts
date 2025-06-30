# A-Machine-Learning-Approach-to-Analyzing-Development-Aid-in-Conflict-and-Peace-Contexts

## Project Overview

This project implements a **reproducible machine learning pipeline** to semantically classify peace-related development aid projects using **zero-shot text classification**. The main objective is to map CRS purpose code 15220: “Civilian peace-building, conflict prevention and resolution” to specific **SDG targets** (e.g., 16.1, 16.2, 5.2) using BART-large-MNLI and to analyze how aligned these projects are with conflict indicators across countries.

---

## Folder Structure
project/
├── data/                # Raw and processed data (OECD, ACLED, annotations)
│   ├── original/        # Original source files (e.g., downloaded CSVs)
│   ├── processed/       # Cleaned and matched datasets
│   └── validation/      # Labeled samples for evaluation
│
├── scripts/             # Python scripts for classification, evaluation, and analysis
│   ├── classify_bart_zero_shot.py
│   ├── evaluate_against_manual.py
│   ├── correlation_analysis.py
│   └── utils.py
│
├── results/             # Model outputs, evaluation tables, intermediate exports
├── figures/             # Final plots for thesis (maps, scatterplots, etc.)
├── analysis.ipynb       # Jupyter Notebook to run full analysis from start to finish
├── requirements.txt     # Python dependencies
└── README.md            # This documentation file
