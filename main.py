# Bibliotheken importieren
import pandas as pd
import string
import re
from deep_translator import GoogleTranslator
from tqdm import tqdm
from google.colab import drive

main(

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

)
