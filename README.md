# A-Machine-Learning-Approach-to-Analyzing-Development-Aid-in-Conflict-and-Peace-Contexts

## Abstract

Peacebuilding is a central objective of global development, particularly under Sustainable Development Goal (SDG) 16. A key instrument for promoting peace is development aid, yet the extent to which aid flows align with SDG 16 targets remains unclear. While existing aid data provides project-level information, the link between aid activities and SDG 16 has not been comprehensively analyzed. Here, we develop a novel, scalable machine learning framework to map aid projects onto SDG 16 targets. Specifically, we use zero-shot semantic entailment (BART-large-MNLI) to classify aid projects against SDG 16 targets (e.g., \enquote{Significantly reduce all forms of violence and related death rates everywhere} (SDG 16.1)). We apply our framework to over 68,000 aid projects, with USD 36.7 billion in disbursements, to assess how donors support peacebuilding objectives. Our analysis reveals that donor funding is heavily concentrated on a narrow subset of SDG 16 targets and countries such as Afghanistan, while key areas such as \enquote{End abuse, exploitation, trafficking and all forms of violence against and torture of children} (SDG 16.2) remain underfunded. This points to potential disparities in how aid is allocated across peacebuilding priorities. Our framework offers a scalable approach for tracking aid contributions to SDG 16. By enhancing transparency and supporting evidence-based decision-making, it can guide more targeted and equitable funding supporting peacebuilding efforts.

## 🧭 Project Objective

Global development efforts increasingly emphasize alignment with policy goals. However,
existing aid reporting systems, most notably the Organisation for Economic Co-operation
and Development (OECD) Creditor Reporting System (CRS), offer only limited insight
into the actual content of funded projects. This is particularly problematic in the domain
of peacebuilding, where overly broad purpose codes subsume a wide range of divergent
initiatives without sufficient differentiation. Furthermore, despite the presence of global
policy frameworks like the Sustainable Development Goals (SDGs), CRS data remains poorly
aligned with their target-level categories. The current SDG labelling practice is applied
inconsistently and lacks semantic precision, creating a fundamental gap in the evaluation
of aid alignment. This thesis aims to close this gap by developing a scalable classification
framework that semantically maps aid project descriptions to specific SDG sub-targets. The
method leverages a zero-shot classification approach using the BART-large-MNLI model,
which evaluates semantic entailment between policy prompts and project texts. A total of
over 68,000 projects were classified to SDG targets 16.1–16.4, 16.6, 16.8 and 5.2. The results
demonstrate that the proposed framework enables substantially greater thematic resolution
compared to the existing CRS purpose codes and SDG labels. The framework enables more
accurate mappings between disbursement flows and normative goals, and can be part of
more robust monitoring and evaluation systems in international cooperation.

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
