import pandas as pd
import re
import ast
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Using entity resolution only for the labels that matter because only they have references across articles. 
#Numbers,Stats and Dates don't.

mainLabels = {
    'PERSON': 'Players and officials',
    'ORG': 'Teams and cricket boards',
    'GPE': 'Countries and cities',
    'LOC': 'Stadiums and venues',
    'EVENT': 'Tournaments',
}


dataframe = pd.read_csv("CorefTest.csv")
dataframe["NER"] = dataframe["NER"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
entity_dict = {}  

for entities in dataframe["NER"]:
    for entity, label in entities:
        label = label.strip()
       
        if label in mainLabels:
            norm_entity = entity.lower().strip()
            if norm_entity not in entity_dict:
                entity_dict[norm_entity] = (entity, label)

unique_entities = list(entity_dict.keys())
print(f"Extracted {len(unique_entities)} unique entities.")


def normalize_entity(entity):
    entity = re.sub(r'[^\w\s]', '', entity.lower().strip())
    return entity

normalized_entities = [normalize_entity(ent) for ent in unique_entities]


vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
tfidf_matrix = vectorizer.fit_transform(normalized_entities)
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


alias_groups = defaultdict(set)
threshold = 0.75  

for i in range(len(unique_entities)):
    for j in range(i + 1, len(unique_entities)):
        label_i = entity_dict[unique_entities[i]][1]
        label_j = entity_dict[unique_entities[j]][1]

        if label_i == label_j and cosine_sim_matrix[i, j] > threshold:
            alias_groups[unique_entities[i]].add(unique_entities[j])
            alias_groups[unique_entities[j]].add(unique_entities[i])


final_aliases = {}
for main_entity, aliases in alias_groups.items():
    full_aliases = {entity_dict[ent][0] for ent in aliases if ent in entity_dict}
    full_aliases.add(entity_dict[main_entity][0])
    final_aliases[entity_dict[main_entity][0]] = list(full_aliases)


def resolve_aliases(entity):

    if re.match(r'^\d+$', entity):  # numbers
        return [entity]
    if re.match(r'^\d{1,2}(st|nd|rd|th)?$', entity):  # ordinals
        return [entity]
    if re.match(r'^\d{4}$', entity):  # Years 
        return [entity]
    if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', entity.lower()):
        return [entity]  # Month-based dates - "February 19"
    return final_aliases.get(entity, [entity])


dataframe["EntityResolution"] = dataframe["NER"].apply(
    lambda ents: [{ent[0]: resolve_aliases(ent[0])} for ent in ents]
)

dataframe.to_csv("entityResolved.csv", index=False)

