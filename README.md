# A-Machine-Learning-Approach-to-Analyzing-Development-Aid-in-Conflict-and-Peace-Contexts

🌍 SDG-Aligned Classification of Peacebuilding Aid Projects

⸻

🧭 Ziel des Projekts

Dieses Projekt zielt darauf ab, einen skalierbaren Zero-Shot-Ansatz zur inhaltlichen Kategorisierung von Entwicklungshilfeprojekten im Bereich Friedensförderung und Konfliktprävention zu entwickeln.
Mithilfe eines Large Language Models (BART-large-MNLI) werden Textbeschreibungen aus dem OECD-CRS-Datensatz automatisch sieben konkreten SDG-Teilzielen (u. a. SDG 16.1, 16.2, 16.6, 5.2) zugeordnet. Dadurch wird eine präzisere, themenspezifische Analyse von Förderprioritäten ermöglicht.

⸻

🛠️ Code-Ausführung

1. Umgebung vorbereiten

Installiere alle notwendigen Python-Pakete:
pip install -r requirements.txt


Öffne und führe das Notebook analysis.ipynb aus. Es enthält alle Schritte:
	•	📥 Datenimport & Vorverarbeitung
	•	🧠 Zero-Shot-Klassifikation
	•	📈 Korrelation mit Konfliktdaten (ACLED)
	•	📊 Visualisierung & Export der Ergebnisse

Alternativ können einzelne Schritte über Skripte im Ordner scripts/ ausgeführt werden.

⸻

📂 Verwendete Daten
	•	OECD-CRS (Creditor Reporting System) für Disbursement-Daten von Friedensprojekten (2023)
	•	ACLED Conflict Index für konfliktbezogene Länderscores (2024)
	•	Manuell annotierte Labels für 300 Textbeschreibungen zur Evaluierung

📊 Wichtigste Outputs
	•	📄 SDG-Labels pro Projekt (Zero-Shot klassifiziert)
	•	🗺️ Geografische Verteilung pro SDG-Ziel (figures/Map_*.pdf)
	•	📉 Scatterplots zu Aid–Conflict-Zusammenhängen (figures/Scatter_*.pdf)
	•	📋 Evaluierungstabellen (Precision, Recall, F1-Scores)
	•	🤖 Vergleich mit GPT-4 Labels (Benchmarking gegen manuelle Annotation)
