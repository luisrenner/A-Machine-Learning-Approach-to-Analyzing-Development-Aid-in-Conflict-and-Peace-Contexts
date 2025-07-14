# A-Machine-Learning-Approach-to-Analyzing-Development-Aid-in-Conflict-and-Peace-Contexts

## Abstract

Peacebuilding is a central objective of global development, particularly under Sustainable Development Goal (SDG) 16. A key instrument for promoting peace is development aid, yet the extent to which aid flows align with SDG 16 targets remains unclear. While existing aid data provides project-level information, the link between aid activities and SDG 16 has not been comprehensively analyzed. Here, we develop a novel, scalable machine learning framework to map aid projects onto SDG 16 targets. Specifically, we use zero-shot semantic entailment (BART-large-MNLI) to classify aid projects against SDG 16 targets (e.g., \enquote{Significantly reduce all forms of violence and related death rates everywhere} (SDG 16.1)). We apply our framework to over 68,000 aid projects, with USD 36.7 billion in disbursements, to assess how donors support peacebuilding objectives. Our analysis reveals that donor funding is heavily concentrated on a narrow subset of SDG 16 targets and countries such as Afghanistan, while key areas such as \enquote{End abuse, exploitation, trafficking and all forms of violence against and torture of children} (SDG 16.2) remain underfunded. This points to potential disparities in how aid is allocated across peacebuilding priorities. Our framework offers a scalable approach for tracking aid contributions to SDG 16. By enhancing transparency and supporting evidence-based decision-making, it can guide more targeted and equitable funding supporting peacebuilding efforts.

## Getting started

### 1. Clone the repository:

git clone https://github.com/forsterkerstin/monitoring-public-health.git

Clone the repository:

git clone https://github.com/forsterkerstin/monitoring-public-health.git
Install dependencies:

pip install -r requirements.txt
Download data: Download the required data folder from https://osf.io/ce3q2/ and place it at root level, as specified in the folder structure below.

Run scripts: The data and plots folders will be automatically created at runtime as needed by the scripts.

Run the scripts in the following order: preprocessing 
→
 classification 
→
 visualization. You can run main.py to execute the core analysis scripts without incurring API costs.

Preprocessing:
01_detect_language.py
02_translate_texts.py
03_combine_aid_funding_datasets.py
Classification:
04_llm_classification.py
05_keyword_classification.py
06_add_classification_to_aid_data.py
Visualization:
07_aid_funding_plots.py
08_correlation_and_funding_disparities_plots.py
09_comparing_classification_methods.py
Note: main.py is provided to run only the visualization and analysis scripts, as the translation and classification scripts require paid API keys.



## 🛠️ How to Run the Code

### 1. Set up the environment

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 2. Run the main analysis

Open and run the notebook `analysis.ipynb`, which includes all steps:

- Data import & preprocessing  
- Zero-shot classification  
- Correlation with conflict data (ACLED)  
- Visualization & export of results  

Alternatively, individual steps can be executed using scripts located in the `scripts/` folder.

## 📂 Data Used

- OECD-CRS (Creditor Reporting System) for disbursement data on peace-related projects (2023)
- ACLED Conflict Index for conflict scores at the country level (2024)
- Manually annotated labels for 300 project descriptions used for validation

All data is organized within the `data/` directory:

- `data/original/`: Raw data sources
- `data/processed/`: Preprocessed files
- `data/validation/`: Gold-standard labels for model evaluation

## 📊 Key Outputs

The most important outputs of this project are:

- SDG labels per project (automatically classified)
- Visualizations:
  - Geographic distributions per SDG target (`figures/Map_*.pdf`)
  - Scatterplots on aid–conflict relationships (`figures/Scatter_*.pdf`)
- Evaluation tables with precision, recall, and F1-scores
- Comparison with GPT-4 classifications (benchmarking against manual annotations)
