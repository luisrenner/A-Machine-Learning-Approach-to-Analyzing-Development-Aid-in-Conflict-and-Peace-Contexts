# Bibliotheken importieren
import pandas as pd
import string
import re
from deep_translator import GoogleTranslator
from tqdm import tqdm
from transformers import pipeline
from datasets import Dataset
from tqdm.auto import tqdm
import torch
import numpy as np

# CSV laden
df = pd.read_csv("Civilian Peacebuilding Dataset.csv")

# Texte kombinieren
df['FullText'] = df[['ProjectTitle', 'ShortDescription', 'LongDescription']].fillna('').agg(' '.join, axis=1)

# Übersetzung ins Englische (via Google Translate API)
translated = []
translator = GoogleTranslator(source='auto', target='en')

for text in tqdm(df['FullText'].tolist()):
  try:
    translated.append(translator.translate(text))
  except Exception as e:
    translated.append(text)  # falls Fehler, original belassen

df['TranslatedText'] = translated

# Speichern
df.to_csv("Civilian Peacebuilding Translated.csv", index=False)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# CSV laden
df = pd.read_csv("Civilian Peacebuilding Translated.csv")
df = df[df['TranslatedText'].notna()].reset_index(drop=True)

# MULTI-PROMPT-Hypothesen für jedes SDG
sdg_prompts = {
    "Significantly reduce all forms of violence and related death rates everywhere (SDG 16.1)": [
        "This project contributes to reducing violence.",
        "This project prevents armed conflict or physical harm.",
        "This project aims to reduce violent-related deaths."
    ],
    "End abuse, exploitation, trafficking and all forms of violence against and torture of children (SDG 16.2)": [
        "This project protects children from abuse or exploitation.",
        "This project fights human trafficking or sexual violence.",
        "This project prevents neglect and mistreatment of children."
    ],
    "Promote the rule of law at the national and international levels and ensure equal access to justice for all (SDG 16.3)": [
        "This project promotes the rule of law and justice.",
        "This project improves access to fair legal systems.",
        "This project strengthens judicial institutions."
    ],
    "By 2030, significantly reduce illicit financial and arms flows, strengthen the recovery and return of stolen assets and combat all forms of organized crime (SDG 16.4)": [
        "This project fights organized crime or money laundering.",
        "This project addresses illicit arms trafficking.",
        "This project prevents illegal financial flows."
    ],
    "Develop effective, accountable and transparent institutions at all levels (SDG 16.6)": [
        "This project supports institutional reforms.",
        "This project strengthens public administration and governance structures.",
        "This project promotes transparency and accountability in governance."
    ],
    "Broaden and strengthen the participation of developing countries in the institutions of global governance (SDG 16.8)": [
        "This project enhances the participation of developing countries in global governance.",
        "This project builds capacity for engagement in international institutions.",
        "This project supports inclusive representation in multilateral decision-making."
    ],
    "Eliminate all forms of violence against all women and girls in the public and private spheres, including trafficking and sexual and other types of exploitation (SDG 5.2)": [
        "This project helps eliminate violence against women and girls.",
        "This project addresses gender-based violence.",
        "This project supports female survivors of abuse."
    ]
}

# Klassifikationsdurchlauf mit Mean Aggregation
results_dict = {k: [] for k in sdg_prompts.keys()}
top_sdgs = []
top_scores = []

for i, text in tqdm(enumerate(df["TranslatedText"]), total=len(df), desc="🔍 Classifying"):
    sdg_scores = {}
    for sdg, prompts in sdg_prompts.items():
        scores = []
        for hyp in prompts:
            res = classifier(text, [hyp], multi_label=False)
            scores.append(res["scores"][0])
        sdg_scores[sdg] = np.mean(scores)  # <- mean aggregation
        results_dict[sdg].append(sdg_scores[sdg])

    # Top SDG
    top_sdg = max(sdg_scores, key=sdg_scores.get)
    top_sdgs.append(top_sdg)
    top_scores.append(sdg_scores[top_sdg])

# DataFrame zusammenführen
for k in results_dict:
    df[k] = results_dict[k]
df["Top_SDG"] = top_sdgs
df["Top_Score"] = top_scores

# Relevante SDGs (Score > 0.4)
def assign_sdg(row):
    return [sdg for sdg in sdg_prompts if row[sdg] > 0.4]
df["Assigned_SDGs"] = df.apply(assign_sdg, axis=1)

# Speichern
df.to_csv("Civilian Peacebuilding SDG-Mapped.csv", index=False)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# CSV laden
df = pd.read_csv("Civilian Peacebuilding Translated.csv")
df = df[df['TranslatedText'].notna()].reset_index(drop=True)

# SINGLE-PROMPT-Hypothesen für jedes SDG
sdg_prompts = {
    "Significantly reduce all forms of violence and related death rates everywhere (SDG 16.1)": [
        "This project aims to significantly reduce all forms of violence and related death rates everywhere."
    ],
    "End abuse, exploitation, trafficking and all forms of violence against and torture of children (SDG 16.2)": [
        "This project aims to end abuse, exploitation, trafficking and all forms of violence against and torture of children."
    ],
    "Promote the rule of law at the national and international levels and ensure equal access to justice for all (SDG 16.3)": [
        "This project aims to promote the rule of law at the national and international levels and ensure equal access to justice for all."
    ],
    "By 2030, significantly reduce illicit financial and arms flows, strengthen the recovery and return of stolen assets and combat all forms of organized crime (SDG 16.4)": [
        "This project aims to significantly reduce illicit financial and arms flows, strengthen the recovery and return of stolen assets and combat all forms of organized crime."
    ],
    "Eliminate all forms of violence against all women and girls in the public and private spheres, including trafficking and sexual and other types of exploitation (SDG 5.2)": [
        "This project aims to eliminate all forms of violence against all women and girls in the public and private spheres, including trafficking and sexual and other types of exploitation."
    ],
    "Develop effective, accountable and transparent institutions at all levels (SDG 16.6)": [
        "This project aims to develop effective, accountable and transparent institutions at all levels."
    ],
    "Broaden and strengthen the participation of developing countries in the institutions of global governance (SDG 16.8)": [
        "This project aims to broaden and strengthen the participation of developing countries in the institutions of global governance."
    ]
}

# Klassifikationsdurchlauf mit Mean Aggregation
results_dict = {k: [] for k in sdg_prompts.keys()}
top_sdgs = []
top_scores = []

for i, text in tqdm(enumerate(df["TranslatedText"]), total=len(df), desc="🔍 Classifying"):
    sdg_scores = {}
    for sdg, prompts in sdg_prompts.items():
        scores = []
        for hyp in prompts:
            res = classifier(text, [hyp], multi_label=False)
            scores.append(res["scores"][0])
        sdg_scores[sdg] = np.mean(scores)  # <- mean aggregation
        results_dict[sdg].append(sdg_scores[sdg])

    # Top SDG
    top_sdg = max(sdg_scores, key=sdg_scores.get)
    top_sdgs.append(top_sdg)
    top_scores.append(sdg_scores[top_sdg])

# DataFrame zusammenführen
for k in results_dict:
    df[k] = results_dict[k]
df["Top_SDG"] = top_sdgs
df["Top_Score"] = top_scores

# Relevante SDGs (Score > 0.4)
def assign_sdg(row):
    return [sdg for sdg in sdg_prompts if row[sdg] > 0.4]

df["Assigned_SDGs"] = df.apply(assign_sdg, axis=1)

# Speichern
df.to_csv("Civilian Peacebuilding SDG-Mapped single prompt.csv", index=False)
