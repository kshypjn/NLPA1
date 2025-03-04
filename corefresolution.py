import pandas as pd
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import tqdm
from collections import defaultdict

#Saves the coreference resolutions in CorefTest.csv
dataframe = pd.read_csv('NERtagged.csv')
column = 'content'

print("Loading AllenNLP coreference resolution model...")
predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")
print("Model loaded successfully.")


def resolveCoref(text):
    try:
        prediction = predictor.predict(document=text)
    except Exception as e:
        print(f"Error processing text: {e}")
        return text, {}

    clusters = prediction.get('clusters', [])
    tokens = prediction.get('document', [])
    resolved_tokens = tokens.copy()

    replaced_indices = set()

    entity_mentions = defaultdict(lambda: {"count": 0, "positions": []})

    for cluster in clusters:
        main_mention = cluster[0]
        main_mention_text = ' '.join(tokens[main_mention[0]:main_mention[1] + 1])

        entity_mentions[main_mention_text]["count"] += 1
        entity_mentions[main_mention_text]["positions"].append(main_mention[0])

        for mention in sorted(cluster[1:], key=lambda x: -x[0]):
            start, end = mention

            if any(i in replaced_indices for i in range(start, end + 1)):
                continue

            mention_span = tokens[start:end + 1]
            entity_mentions[main_mention_text]["count"] += 1
            entity_mentions[main_mention_text]["positions"].append(start)

            if mention_span[-1] == "'s" or mention_span[-1].lower() in ["his", "her", "its"]:
                main_mention_text_possessive = main_mention_text + "'s"
                resolved_tokens[start] = main_mention_text_possessive
                for i in range(start + 1, end + 1):
                    resolved_tokens[i] = ''
                    replaced_indices.add(i)
            else:
                resolved_tokens[start] = main_mention_text
                for i in range(start + 1, end + 1):
                    resolved_tokens[i] = ''
                    replaced_indices.add(i)
            replaced_indices.add(start)

    resolved_text = ' '.join([token for token in resolved_tokens if token.strip()])
    return resolved_text, dict(entity_mentions)


tqdm.tqdm.pandas()
results = dataframe[column].progress_apply(resolveCoref)

dataframe['Coref1'] = results.apply(lambda x: x[0])
dataframe['Coref2'] = results.apply(lambda x: x[1])
#Coref1 replaces the words that have coreferences like 'Kohli' is replaced by 'Virat Kohli'
#Coref2 gives the count of a coreference and its position in a JSON like format : "{'The ICC Champions Trophy': {'count': 2, 'positions': [0, 35]}}"
dataframe.to_csv('CorefTest.csv', index=False)
print("Coreference resolution!")

