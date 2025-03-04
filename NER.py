
import spacy
import pandas as pd
#trf because it does a better job at NER than sm or md. tried it with the other 2.
model = spacy.load("en_core_web_trf")  

# had to decide so many labels because of variety of data 
cricLabels = {
    'PERSON': 'Players and officials',
    'ORG': 'Cricket Boards',
    'GPE': 'Teams, Countries and cities',
    'LOC': 'Stadiums and venues',
    'EVENT': 'Tournaments',
    'DATE': 'Dates',
    'CARDINAL': 'Numerical values',
    'ORDINAL': 'Rankings'
}

dataframe = pd.read_csv("POStagged.csv")


def extract(text):
    doc = model(text)
    return [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in cricLabels]


dataframe['NER'] = dataframe['content'].apply(extract)
dataframe.to_csv('NERtagged.csv', index=False)
print("done")
