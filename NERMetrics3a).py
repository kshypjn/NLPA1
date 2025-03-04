import pandas as pd
import json
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def manualAnnotations(annotations_str):
    try:
        annotations = json.loads(annotations_str.replace('""', '"'))
        entities = [(ann["start"], ann["end"], ann["labels"][0]) for ann in annotations]
        return entities
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []

def formatChange(csv_path):
    dataframe = pd.read_csv(csv_path)
    converted_rows = []

    for _, row in dataframe.iterrows():
        text = row["text"]
        annotations = manualAnnotations(row["label"])

        for start, end, label in annotations:
            word = text[start:end]
            converted_rows.append({"word": word, "ner": label})

    converteddataframe = pd.DataFrame(converted_rows)
    converteddataframe.to_csv("converted_NER.csv", index=False)
    print("Converted data saved to converted_NER.csv")

    return converted_rows

def evalNER(manual_data, tagged_csv_path):
    dataframetagged = pd.read_csv(tagged_csv_path).head(30)

    taggedentities = []
    for _, row in dataframetagged.iterrows():
        ner_list = eval(row["NER"])
        for entity, label in ner_list:
            taggedentities.append({"word": entity, "ner": label})

    y_true = []
    y_pred = []
    mismatches = []

    with open("NER_metrics.txt", "w") as f:
        for manual_entity in manual_data:
            word = manual_entity["word"]
            true_label = manual_entity["ner"]

            matched = next((item["ner"] for item in taggedentities if item["word"] == word), None)

            if matched:
                y_true.append(true_label)
                y_pred.append(matched)
            else:
                mismatches.append(word)

        if mismatches:
            f.write(f"Mismatched entities (not found in predictions): {mismatches}\n")

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

csv_path = "NERManual.csv"
tagged_csv_path = "NERtagged.csv"
manual_data = formatChange(csv_path)
evalNER(manual_data, tagged_csv_path)
