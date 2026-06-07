import json
import csv
import random
import os

random.seed(42)

DIR = os.path.dirname(__file__)

# Load English data first for reference
english_path = os.path.join(DIR, "english_truthful_qa.json")
try:
    with open(english_path, "r", encoding="utf-8") as f:
        english_data = json.load(f)
except Exception as e:
    print(f"Failed to read {english_path}: {e}")
    english_data = []

FILES = [
    "english_truthful_qa.json",
    "hindi_truthful_qa.json",
    "bengali_truthful_qa.json",
]

for fname in FILES:
    path = os.path.join(DIR, fname)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        continue

    n = min(50, len(data))
    if len(data) == 0:
        print(f"No entries in {fname}, skipping")
        continue

    # Get indices to sample
    indices = random.sample(range(len(data)), n) if len(data) >= n else range(len(data))
    samples = [data[i] for i in indices]

    csv_name = fname.replace("truthful_qa.json", "samples.csv")
    csv_path = os.path.join(DIR, csv_name)

    def list_to_safe_str(lst):
        if isinstance(lst, list):
            # filter out None and convert items to str
            return " || ".join(str(x) for x in lst if x is not None)
        return str(lst)

    # Determine fieldnames based on whether we need English columns
    if fname == "english_truthful_qa.json":
        fieldnames = [
            "question",
            "best_answer",
            "correct_answers",
            "incorrect_answers",
            "category",
        ]
    else:
        fieldnames = [
            "question",
            "best_answer",
            "correct_answers",
            "incorrect_answers",
            "category",
            "english_question",
            "english_best_answer",
            "english_correct_answers",
            "english_incorrect_answers",
        ]

    with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for idx, item in zip(indices, samples):
            ca = item.get("correct_answers", [])
            ia = item.get("incorrect_answers", [])

            row = {
                "question": item.get("question", ""),
                "best_answer": item.get("best_answer", ""),
                "correct_answers": list_to_safe_str(ca),
                "incorrect_answers": list_to_safe_str(ia),
                "category": item.get("category", ""),
            }

            # Add English versions for Hindi and Bengali
            if fname != "english_truthful_qa.json" and idx < len(english_data):
                eng_item = english_data[idx]
                row["english_question"] = eng_item.get("question", "")
                row["english_best_answer"] = eng_item.get("best_answer", "")
                row["english_correct_answers"] = list_to_safe_str(
                    eng_item.get("correct_answers", [])
                )
                row["english_incorrect_answers"] = list_to_safe_str(
                    eng_item.get("incorrect_answers", [])
                )

            writer.writerow(row)
    print(f"Wrote {n} samples to {csv_path}")
