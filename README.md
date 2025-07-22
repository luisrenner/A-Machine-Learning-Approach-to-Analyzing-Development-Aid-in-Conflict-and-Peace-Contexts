# A Machine Learning Approach to Analyzing Development Aid in Conflict and Peace Contexts

## Abstract

Peace-building is a central objective of global development, particularly under Sustainable Development Goal (SDG) 16. A key instrument for promoting peace is development aid, yet the extent to which aid flows align with SDG 16 targets remains unclear. While existing aid data provides project-level information, a comprehensive analysis of the link between aid activities and SDG 16 is lacking. Here, we develop a novel, scalable large language model (LLM) framework to map aid projects onto SDG 16 targets. Specifically, we use zero-shot semantic entailment (BART-large-MNLI) to classify aid projects against SDG 16 targets (e.g., SDG 16.1: “Significantly reduce all forms of violence and related death rates everywhere”). We apply our LLM framework to over 68,000 aid projects with USD 36.7 billion in disbursements to assess how donors support peace-building objectives. Our analysis reveals that donor funding is heavily concentrated on a narrow subset of SDG 16 targets and countries such as Afghanistan, while funding for other targets, such as SDG 16.2: “End abuse, exploitation, trafficking and all forms of violence against and torture of children”, is comparatively low. This points to potential disparities in how aid is allocated across peace-building priorities. Our framework offers a scalable approach for tracking aid contributions to SDG 16 and can guide more targeted and equitable funding decisions to support peace-building efforts.

## Getting started

### 1. Clone the repository

git clone https://github.com/luisrenner/A-Machine-Learning-Approach-to-Analyzing-Development-Aid-in-Conflict-and-Peace-Contexts.git

### 2. Install requirements

pip install -r requirements.txt

### 3. Download data

Download the required data folder from https://stats.oecd.org/Index.aspx?DataSetCode=CRS1 and place it at root level, as specified in the folder structure below.

### 4. Run code

The data and plots folders will be automatically created at runtime as needed by the scripts.

Run the scripts in the following order: preprocessing → classification → visualization. 

#### Preprocessing

01_preprocessing.py
02_bart_large_mnli_classification.py
05_keyword_classification.py
06_add_classification_to_aid_data.py
