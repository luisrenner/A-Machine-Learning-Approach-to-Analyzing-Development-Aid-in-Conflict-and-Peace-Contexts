# Imports
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
import geopandas as gpd
import matplotlib.pyplot as plt
from unidecode import unidecode
from matplotlib.colors import Normalize
import seaborn as sns
from scipy.stats import pearsonr
import statsmodels.api as sm

# Filepaths
input_filepath = "data/Civilian Peacebuilding Dataset.csv"
translated_filepath = "data/Civilian Peacebuilding Translated.csv"
mapped_filepath = "data/Civilian Peacebuilding SDG-Mapped.csv"
singlemapped_filepath = "data/Civilian Peacebuilding SDG-Mapped single prompt.csv"
allmapped_filepath = "data/Peace & Conflict SDG Mapped.csv"
output_filepath = f"data/Map_"
worldmap_filepath = "data/Map_total.pdf"
sdgdisbursements_filepath = "data/SDG_disbursements.pdf"
yearscountries_filepath = "data/jahresausgaben_nach_laendern_und_purpose.csv"
purposenamesovertime_filepath = "data/USD_disbursements_(in_millions)_purpose_names_over_time.pdf"
countriesovertime_filepath = "data/USD_disbursements_(in millions)_top_countries_over_time.pdf"
allmapped2023_filepath = "data/Peace & Conflict SDG Mapped 2023.csv"
results2024_filepath = "data/2024 - Results.csv"
aidvsindicator = f"data/Scatter_aid2023_vs_"
aidvsindicatortotal = f"data/Scatter_aid2023_vs_total_score_2024_SDG_"

# Reader
def read_input(filepath):
  return pd.read_csv(filepath)
def read_input_lowmemory(filepath):
  return pd.read_csv(filepath, low_memory=False)

# Concatenate and translate
def concatenate_and_translate():

  # Load CSV
  df = read_input(input_filepath)

  # Combine texts
  df['FullText'] = df[['ProjectTitle', 'ShortDescription', 'LongDescription']].fillna('').agg(' '.join, axis=1)

  # Translate to English (via Google Translate API)
  translated = []
  translator = GoogleTranslator(source='auto', target='en')

  for text in tqdm(df['FullText'].tolist()):
    try:
      translated.append(translator.translate(text))
    except Exception as e:
      translated.append(text)  # falls Fehler, original belassen

  df['TranslatedText'] = translated

  # Save
  df.to_csv(translated_filepath, index=False)

def multi_prompt_mapping():

  classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

  # Load CSV
  df = read_input(translated_filepath)
  df = df[df['TranslatedText'].notna()].reset_index(drop=True)

  # MULTI-PROMPT-Hypotheses for every SDG
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

  # Classification with mean aggregation
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

  # DataFrame
  for k in results_dict:
      df[k] = results_dict[k]
  df["Top_SDG"] = top_sdgs
  df["Top_Score"] = top_scores

  # Relevant SDGs (Score > 0.4)
  def assign_sdg(row):
      return [sdg for sdg in sdg_prompts if row[sdg] > 0.4]
  df["Assigned_SDGs"] = df.apply(assign_sdg, axis=1)

  # Save
  df.to_csv(mapped_filepath, index=False)

def single_prompt_mapping():

  classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

  # Load CSV
  df = read_input(translated_filepath)
  df = df[df['TranslatedText'].notna()].reset_index(drop=True)

  # SINGLE-PROMPT-Hypotheses fore every SDG
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

  # Classification with mean aggregation
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

  # DataFrame
  for k in results_dict:
      df[k] = results_dict[k]
  df["Top_SDG"] = top_sdgs
  df["Top_Score"] = top_scores

  # Relevant SDGs (Score > 0.4)
  def assign_sdg(row):
      return [sdg for sdg in sdg_prompts if row[sdg] > 0.4]

  df["Assigned_SDGs"] = df.apply(assign_sdg, axis=1)

  # Save
  df.to_csv(singlemapped_filepath, index=False)

def worldmapsdg():

  # Load data
  df = read_input_lowmemory(allmapped_filepath)

  # Clean-up recipient-names
  df["RecipientName_clean"] = df["RecipientName"].apply(lambda x: unidecode(str(x).strip().lower()))
  df["USD_Disbursement"] = (
      df["USD_Disbursement"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )

  # Exclude non-country geographies
  non_countries = [
      "africa, regional", "america, regional", "asia, regional", "europe, regional",
      "oceania, regional", "south america, regional", "south asia, regional",
      "western africa, regional", "eastern africa, regional", "southern africa, regional",
      "central asia, regional", "central america, regional", "middle east, regional",
      "middle africa, regional", "caribbean, regional", "caribbean & central america, regional",
      "far east asia, regional", "north of sahara, regional", "south of sahara, regional",
      "south & central asia, regional", "melanesia, regional", "bilateral, unspecified",
      "states ex-yugoslavia unspecified", "tokelau"
  ]
  df = df[~df["RecipientName_clean"].isin(non_countries)].copy()

  # Map country-names
  manual_map = {
      "china (people's republic of)": "china",
      "democratic people's republic of korea": "north korea",
      "democratic republic of the congo": "democratic republic of the congo",
      "cote d'ivoire": "ivory coast",
      "lao people's democratic republic": "laos",
      "syrian arab republic": "syria",
      "viet nam": "vietnam",
      "turkiye": "turkey",
      "micronesia": "micronesia (federated states of)",
      "west bank and gaza strip": "palestine",
      "congo": "republic of the congo",
      "north macedonia": "north macedonia",
      "cabo verde": "cape verde",
      "eswatini": "swaziland",
      "timor-leste": "east timor",
      "sao tome and principe": "são tomé and príncipe",
      "trinidad and tobago": "trinidad and tobago",
      "myanmar": "myanmar",
      "kosovo": "kosovo",
      "republic of moldova": "moldova",
      "bahamas, the": "bahamas",
      "gambia, the": "gambia",
      "venezuela (bolivarian republic of)": "venezuela",
      "iran (islamic republic of)": "iran",
      "russian federation": "russia",
      "bolivia (plurinational state of)": "bolivia",
      "tanzania, united republic of": "tanzania",
      "korea, republic of": "south korea",
      "türkiye": "turkey",
      "serbia": "republic of serbia"
  }

  # Load world-map
  world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
  world["name"] = world["name"].str.strip().str.lower()

  # SDG-labels
  unique_sdg_labels = df["AssignedLabel"].dropna().unique()

  for label in unique_sdg_labels:
      df_label = df[df["AssignedLabel"] == label].copy()
      usd_label_total = df_label["USD_Disbursement"].sum()

      # Map and group
      df_label["MatchName"] = df_label["RecipientName_clean"].replace(manual_map)
      label_data = df_label.groupby("MatchName", as_index=False)["USD_Disbursement"].sum()

      # Merge with world-map
      merged = world.merge(label_data, how="left", left_on="name", right_on="MatchName")

      # Plot
      fig, ax = plt.subplots(figsize=(28, 14))
      norm = Normalize(vmin=0, vmax=merged["USD_Disbursement"].max())

      merged.plot(
          column="USD_Disbursement",
          cmap="Reds",
          linewidth=0.5,
          edgecolor="white",
          ax=ax,
          legend=True,
          norm=norm,
          missing_kwds={
              "color": "white",        # Länder ohne Funding
              "edgecolor": "lightgray" # Dezente Umrandung
          },
          legend_kwds={
              "shrink": 0.6,
              "orientation": "vertical"
          }
      )

      # Set font-size
      colorbar = ax.get_figure().get_axes()[-1]
      colorbar.tick_params(labelsize=30)
      colorbar.set_ylabel("Disbursements (in USD millions)", fontsize=30)

      # Cut map
      ax.set_ylim(-60, 85)
      ax.set_xlim(-150, 150)
      ax.axis("off")

      # Save
      plt.tight_layout()
      output_path = output_filepath+str(label)+".pdf" 
      plt.savefig(output_path, bbox_inches="tight")
      plt.close()

def worldmaptotal():

  # Load data
  df = read_input_lowmemory(allmapped_filepath)

  # Clean-up recipients
  df["RecipientName_clean"] = df["RecipientName"].apply(lambda x: unidecode(str(x).strip().lower()))
  df["USD_Disbursement"] = (
      df["USD_Disbursement"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )

  # Exclude non-country geographies
  non_countries = [
      "africa, regional", "america, regional", "asia, regional", "europe, regional",
      "oceania, regional", "south america, regional", "south asia, regional",
      "western africa, regional", "eastern africa, regional", "southern africa, regional",
      "central asia, regional", "central america, regional", "middle east, regional",
      "middle africa, regional", "caribbean, regional", "caribbean & central america, regional",
      "far east asia, regional", "north of sahara, regional", "south of sahara, regional",
      "south & central asia, regional", "melanesia, regional", "bilateral, unspecified",
      "states ex-yugoslavia unspecified", "tokelau"
  ]
  df = df[~df["RecipientName_clean"].isin(non_countries)].copy()

  # Map country-names
  manual_map = {
      "china (people's republic of)": "china",
      "democratic people's republic of korea": "north korea",
      "democratic republic of the congo": "democratic republic of the congo",
      "cote d'ivoire": "ivory coast",
      "lao people's democratic republic": "laos",
      "syrian arab republic": "syria",
      "viet nam": "vietnam",
      "turkiye": "turkey",
      "micronesia": "micronesia (federated states of)",
      "west bank and gaza strip": "palestine",
      "congo": "republic of the congo",
      "north macedonia": "north macedonia",
      "cabo verde": "cape verde",
      "eswatini": "swaziland",
      "timor-leste": "east timor",
      "sao tome and principe": "são tomé and príncipe",
      "trinidad and tobago": "trinidad and tobago",
      "myanmar": "myanmar",
      "kosovo": "kosovo",
      "republic of moldova": "moldova",
      "bahamas, the": "bahamas",
      "gambia, the": "gambia",
      "venezuela (bolivarian republic of)": "venezuela",
      "iran (islamic republic of)": "iran",
      "russian federation": "russia",
      "bolivia (plurinational state of)": "bolivia",
      "tanzania, united republic of": "tanzania",
      "korea, republic of": "south korea",
      "türkiye": "turkey",
      "serbia": "republic of serbia"
  }

  # Load world-map
  world = gpd.read_file("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson")
  world["name"] = world["name"].str.strip().str.lower()

  # Map and aggregate
  df["MatchName"] = df["RecipientName_clean"].replace(manual_map)
  total_data = df.groupby("MatchName", as_index=False)["USD_Disbursement"].sum()

  # Merge with world-map
  merged = world.merge(total_data, how="left", left_on="name", right_on="MatchName")

  # Plot
  fig, ax = plt.subplots(figsize=(28, 14))
  norm = Normalize(vmin=0, vmax=merged["USD_Disbursement"].max())

  merged.plot(
    column="USD_Disbursement",
    cmap="Reds",
    linewidth=0.5,
    edgecolor="white",
    ax=ax,
    legend=True,
    norm=norm,
    missing_kwds={
        "color": "white",        # Länder ohne Funding
        "edgecolor": "lightgray" # Dezente Umrandung
    },
    legend_kwds={
        "shrink": 0.6,
        "orientation": "horizontal"
    }
  )
  
  # Set font-size
  colorbar = ax.get_figure().get_axes()[-1]
  colorbar.tick_params(labelsize=30)
  colorbar.set_xlabel("Disbursements (in USD millions)", fontsize=30)

  # Cut map
  ax.set_ylim(-60, 85)
  ax.set_xlim(-150, 150)
  ax.axis("off")

  # Save
  plt.tight_layout()
  plt.savefig(worldmap_filepath, bbox_inches="tight")
  plt.close()

def sdgdisbursements():

  # Load data
  df = read_input_lowmemory(allmapped_filepath)
  df["USD_Disbursement"] = (
      df["USD_Disbursement"].astype(str).str.replace(",", "").astype(float)
  )
  df = df[df["AssignedLabel"].notna()].copy()
  df["AssignedLabel"] = df["AssignedLabel"].astype(str)

  # Aggregation
  bar_data = df.groupby("AssignedLabel")["USD_Disbursement"].sum()
  sdg_order = ["16.1", "16.3", "16.8", "16.6", "16.2", "5.2", "16.4"]
  bar_data = bar_data.reindex(sdg_order)

  # Color-blind colors
  color_palette = [
      "#332288",  # dunkelblau
      "#117733",  # grün
      "#88CCEE",  # hellblau
      "#DDCC77",  # sandgelb
      "#555555",  # dunkelgrau
      "#AA4499",  # violett
      "#44AA99",  # türkisgrün
  ]

  # Plot
  fig, ax = plt.subplots(figsize=(12, 6))
  bars = ax.bar([f"SDG {k}" for k in bar_data.index], bar_data.values, color=color_palette)

  # Axis and layout
  ax.set_ylabel("Disbursements (in USD millions)", fontsize=18)
  ax.tick_params(axis='x', labelsize=14)
  ax.tick_params(axis='y', labelsize=14)
  ax.spines[['top', 'right']].set_visible(False)
  ax.grid(axis='y', linestyle='--', alpha=0.5)
  ax.set_axisbelow(True)

  # Export
  plt.tight_layout()
  plt.savefig(sdgdisbursements_filepath)
  plt.show()

def purposenamesovertime():

  # Style and color
  plt.style.use("seaborn-v0_8-whitegrid")
  sns.set_palette("colorblind")

  # Load CSV
  df = read_input(yearscountries_filepath)
  df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

  # Clean contributions
  df["usd_disbursement_in_millions"] = (
      df["usd_disbursement_in_millions"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )

  # Remove micro-cluster
  df = df[df["purpose_name"] != "Conflict, Peace & Security"]

  # Group
  df_grouped = (
      df.groupby(["year", "purpose_name"], as_index=False)["usd_disbursement_in_millions"]
      .sum()
  )

  # Prepare plots
  fig, ax = plt.subplots(figsize=(14, 8))
  years_full = list(range(df_grouped["year"].min(), df_grouped["year"].max() + 1))
  purpose_order = df_grouped.groupby("purpose_name")["year"].min().sort_values().index
  colors = sns.color_palette("colorblind", n_colors=len(purpose_order))

  # Draw lilnes
  for i, purpose in enumerate(purpose_order):
      sub = df_grouped[df_grouped["purpose_name"] == purpose].set_index("year")
      sub = sub.reindex(years_full, fill_value=0).reset_index()
      ax.plot(
          sub["year"],
          sub["usd_disbursement_in_millions"],
          label=purpose,
          linewidth=2.2,
          color=colors[i]
      )

  # Axis
  ax.set_xlabel("Year", fontsize=18)
  ax.set_ylabel("Disbursements (in USD millions)", fontsize=18)
  ax.tick_params(axis='both', labelsize=16)

  # Legend
  legend = ax.legend(
      title="Purpose",
      title_fontsize=18,
      fontsize=16,
      loc="upper left",
      frameon=True,
      framealpha=0.95,
      facecolor="white",
      edgecolor="gray"
  )

  # Export
  plt.tight_layout()
  plt.savefig(purposenamesovertime_filepath, bbox_inches="tight")
  plt.show()

def countriesovertime():

  # Style
  plt.style.use("seaborn-v0_8-whitegrid")
  sns.set_palette("colorblind")

  # Read data
  df = read_input(yearscountries_filepath)

  # Colum names
  df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

  # Clean-up contributions
  df["usd_disbursement_in_millions"] = (
      df["usd_disbursement_in_millions"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )

  # Countries
  countries = [
      "Afghanistan", "Iraq", "Colombia", "Ukraine", "Democratic Republic of the Congo", "Syrian Arab Republic", "Sudan", "Somalia"
  ]

  df_filtered = df[df["recipient_name_(en)"].isin(countries)]

  # Group
  df_grouped = (
      df_filtered.groupby(["year", "recipient_name_(en)"], as_index=False)
      ["usd_disbursement_in_millions"].sum()
  )

  # Prepare plot
  fig, ax = plt.subplots(figsize=(14, 8))
  years_full = list(range(1992, df_grouped["year"].max() + 1))
  country_order = df_grouped.groupby("recipient_name_(en)")["usd_disbursement_in_millions"].sum().sort_values(ascending=False).index
  colors = sns.color_palette("colorblind", n_colors=len(country_order))

  # Plot
  for i, country in enumerate(country_order):
      sub = df_grouped[df_grouped["recipient_name_(en)"] == country].set_index("year")
      sub = sub.reindex(years_full, fill_value=0).reset_index()
      ax.plot(
          sub["year"],
          sub["usd_disbursement_in_millions"],
          label=country,
          linewidth=2.2,
          color=colors[i]
      )

  # Axis
  ax.set_xlabel("Year", fontsize=18)
  ax.set_ylabel("USD Disbursements in Millions", fontsize=18)
  ax.tick_params(axis='both', labelsize=16)

  # Legend
  ax.legend(
      title="Country",
      title_fontsize=18,
      fontsize=16,
      loc="upper left",
      frameon=True,
      facecolor="white",
      edgecolor="gray"
  )

  # Export
  plt.tight_layout()
  plt.savefig(countriesovertime_filepath, bbox_inches="tight")
  plt.show()

def aidvsindicators():

  # Style
  plt.style.use("seaborn-v0_8-whitegrid")
  sns.set_palette("colorblind")

  # Load and aggregate aid-data
  df_aid = read_input_lowmemory(allmapped2023_filepath)
  df_aid["USD_Disbursement"] = (
      df_aid["USD_Disbursement"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )
  df_aid_grouped = (
      df_aid.groupby("RecipientName", as_index=False)["USD_Disbursement"]
      .sum()
      .rename(columns={"RecipientName": "Country", "USD_Disbursement": "Aid_2023"})
  )

  # Load conflict-data
  df_conflict = read_input_lowmemory(results2024_filepath)
  conflict_cols = ["Country", "Total Score", "Deadliness Value", "Diffusion Value",
                   "Danger Value", "Fragmentation Value"]
  df_conflict = df_conflict[conflict_cols].copy()
  df_conflict = df_conflict.rename(columns=lambda x: x.replace(" ", "_"))

  # Merge
  df_merged = pd.merge(df_aid_grouped, df_conflict, on="Country", how="inner")

  # Scatterplots
  indicators = ["Total_Score", "Deadliness_Value", "Diffusion_Value", "Danger_Value", "Fragmentation_Value"]

  for indicator in indicators:
      plt.figure(figsize=(6, 4))

      # Plot with regression
      sns.regplot(data=df_merged, x="Aid_2023", y=indicator, scatter_kws={"alpha": 0.4}, line_kws={"color": "red"})

      # Correlation
      corr_value = df_merged["Aid_2023"].corr(df_merged[indicator])
      plt.text(0.05, 0.95, f"Correlation: {corr_value:.2f}", transform=plt.gca().transAxes,
               fontsize=12, verticalalignment="top", weight="bold")

      plt.xlabel("Disbursements (in USD millions) 2023")
      plt.ylabel(indicator.replace("_", " ").capitalize() + " 2024")
      plt.tight_layout()

      # Save
      indicator_lower = indicator.lower()
      filename = aidvsindicator+indicator_lower+"_2024.pdf"
      plt.savefig(filename)

      plt.show()

def aidvsindicatorstotal():

  # Style
  plt.style.use("seaborn-v0_8-whitegrid")
  sns.set_palette("colorblind")

  # Load aid-data
  df_aid = read_input(allmapped2023_filepath)
  df_aid["USD_Disbursement"] = (
      df_aid["USD_Disbursement"]
      .astype(str)
      .str.replace(",", "")
      .astype(float)
  )

  # Group country and SDG
  df_sdg_grouped = (
      df_aid.groupby(["RecipientName", "AssignedLabel"], as_index=False)["USD_Disbursement"]
      .sum()
      .rename(columns={"RecipientName": "Country", "USD_Disbursement": "Aid_2023", "AssignedLabel": "SDG"})
  )

  # Load conflict-data
  df_conflict = read_input_lowmemory(results2024_filepath)
  df_conflict = df_conflict[["Country", "Total Score"]].copy()
  df_conflict = df_conflict.rename(columns=lambda x: x.replace(" ", "_"))

  # Merge
  df_merged = pd.merge(df_sdg_grouped, df_conflict, on="Country", how="inner")

  # Scatterplots
  for target in sorted(df_merged["SDG"].dropna().unique()):
      sub = df_merged[df_merged["SDG"] == target].copy()

      # Check data
      if len(sub) < 5 or sub["Aid_2023"].sum() == 0:
          continue

      # Regression
      X = sub["Aid_2023"]
      y = sub["Total_Score"]
      X_const = sm.add_constant(X)
      model = sm.OLS(y, X_const).fit()
      pred_summary = model.get_prediction(X_const).summary_frame(alpha=0.05)

      # Plot
      plt.figure(figsize=(6, 4))
      plt.scatter(X, y, alpha=0.6)

      # Regression
      sort_idx = X.argsort()
      x_sorted = X.iloc[sort_idx]
      mean = pred_summary["mean"].iloc[sort_idx]
      ci_low = pred_summary["mean_ci_lower"].iloc[sort_idx]
      ci_up = pred_summary["mean_ci_upper"].iloc[sort_idx]

      plt.plot(x_sorted, mean, color="red")
      plt.fill_between(x_sorted, ci_low, ci_up, color="red", alpha=0.2)

      # Stastic annotations
      corr, p = pearsonr(X, y)
      r2 = model.rsquared_adj
      plt.text(
        0.05, 0.95,
        f"Correlation: {corr:.2f}\n$p$-value: {p:.3f}\nAdj. $R^2$: {r2:.2f}",
        transform=plt.gca().transAxes,
        fontsize=11, verticalalignment="top", weight="bold"
      )

      # Axis and layout
      plt.xlabel("Disbursements (in USD millions) 2023")
      plt.ylabel("Total conflict score 2024")
      plt.tight_layout()

      # Save
      safe_target = str(target).replace(".", "_")
      out_path = aidvsindicatortotal+safe_target+".pdf"
      plt.savefig(out_path)
      plt.show()

#Execution
concatenate_and_translate()
multi_prompt_mapping()
single_prompt_mapping()
worldmapsdg()
worldmaptotal()
sdgdisbursements()
purposenamesovertime()
countriesovertime()
aidvsindicators()
aidvsindicatorstotal()
