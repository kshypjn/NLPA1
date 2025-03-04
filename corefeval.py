import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def manualAnnotation(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    corefClusters = []
    for entry in data:
        if "annotations" not in entry or not entry["annotations"]:
            continue
        
        annotations = entry["annotations"][0]["result"]
        clusters = set()
        for annotation in annotations:
            if "value" in annotation and "start" in annotation["value"] and "end" in annotation["value"]:
                start = annotation["value"]["start"]
                end = annotation["value"]["end"]
                text = annotation["value"]["text"]
                clusters.add((start, end, text))
        corefClusters.append(clusters)
    return corefClusters

def loadModel(csv_path, column_name="Coref2"):
    dataframe = pd.read_csv(csv_path)
    corefClusters = []
    
    for row in dataframe[column_name]:
        try:
            coref_data = json.loads(row.replace("'", "\""))
            clusters = set()
            for entity, details in coref_data.items():
                for pos in details.get("positions", []):
                    clusters.add((pos, pos + len(entity), entity))
            corefClusters.append(clusters)
        except json.JSONDecodeError:
            corefClusters.append(set())
    return corefClusters

def compute(manual_clusters, model_clusters):
    y_true = []
    y_pred = []
    
    for manual, model in zip(manual_clusters, model_clusters):
        all_mentions = manual.union(model)
        
        y_true.extend([1 if mention in manual else 0 for mention in all_mentions])
        y_pred.extend([1 if mention in model else 0 for mention in all_mentions])
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return precision, recall, f1, accuracy


manual_annotations_path = "corefManual.json"
model_predictions_path = "coref2_extracted.csv"

manual_corefClusters = manualAnnotation(manual_annotations_path)[:15]
model_corefClusters = loadModel(model_predictions_path)[:15]


precision, recall, f1, accuracy = compute(manual_corefClusters, model_corefClusters)

with open("Coref_metrics.txt", "w") as f:
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-score: {f1:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")


