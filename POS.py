import spacy
import pandas as pd
# Saves to POSTagged.csv
dataframe = pd.read_csv("CTScraped.csv")
print(dataframe.head())

model = spacy.load("en_core_web_sm")
def posTagger(text):
    doc = model(text) 
    return [(token.text, token.pos_) for token in doc] 

dataframe["POS_Tags"] = dataframe["content"].apply(posTagger)
dataframe.to_csv("POStagged.csv", index=False)
