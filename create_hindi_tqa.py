import json
from datasets import load_dataset
from googletrans import Translator
import time
from tqdm import tqdm
import requests
import numpy as np


def translate_with_google(text, target_lang="hi"):
    try:
        translator = Translator()
        result = translator.translate(text, dest=target_lang)
        return result.text
    except Exception as _:
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={text}"
            response = requests.get(url)
            if response.status_code == 200:
                translated = "".join([item[0] for item in response.json()[0]])
                return translated
        except Exception as e:
            print(f"Failed to translate: {text}")
            print(f"Error: {e}")
            return text


def translate_with_model(text, model_name="sarvamai/sarvam-1"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    prompt = f"Translate the following English text to Hindi, provide only the translation: {text}"
    conversation = [{"role": "user", "content": prompt}]

    inputs = tokenizer.apply_chat_template(
        conversation, return_tensors="pt", add_generation_prompt=True
    )
    inputs = inputs.cuda()

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
        )

    translation = tokenizer.decode(
        outputs[0, inputs.shape[-1] :], skip_special_tokens=True
    )
    return translation.strip()


def create_hindi_tqa():
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    hindi_dataset = []

    for i, item in enumerate(tqdm(dataset)):
        hindi_question = translate_with_google(item["question"])
        hindi_best_answer = translate_with_google(item["best_answer"])
        time.sleep(0.1)

        hindi_correct_answers = []
        for answer in item["correct_answers"]:
            translated = translate_with_google(answer)
            hindi_correct_answers.append(translated)
        time.sleep(0.1)

        hindi_incorrect_answers = []
        for answer in item["incorrect_answers"]:
            translated = translate_with_google(answer)
            hindi_incorrect_answers.append(translated)
        time.sleep(0.1)

        hindi_item = {
            "question": hindi_question,
            "best_answer": hindi_best_answer,
            "correct_answers": hindi_correct_answers,
            "incorrect_answers": hindi_incorrect_answers,
            "category": item["category"],
        }
        if "question_id" in item:
            hindi_item["question_id"] = item["question_id"]

        hindi_dataset.append(hindi_item)

        if (i + 1) % 100 == 0:
            with open(
                f"hindi_truthful_qa_progress_{i + 1}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(hindi_dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved progress: {i + 1} items translated")

    with open("hindi_truthful_qa.json", "w", encoding="utf-8") as f:
        json.dump(hindi_dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(hindi_dataset)} translated items to hindi_truthful_qa.json")

    print("\nVerifying dataset structure...")
    print(f"Original dataset fields: {list(dataset[0].keys())}")
    print(f"Hindi dataset fields: {list(hindi_dataset[0].keys())}")

    required_fields = [
        "question",
        "best_answer",
        "correct_answers",
        "incorrect_answers",
    ]
    missing_fields = [
        field for field in required_fields if field not in hindi_dataset[0]
    ]
    if missing_fields:
        print(f"Missing fields: {missing_fields}")
    else:
        print("All required fields present!")


def create_hindi_indices():
    eng_index = np.load("data_indices/data_index_tqa.npy")
    eng_exemplar_index = np.load("data_indices/exemplar_idx_tqa.npy")

    np.save("data_indices/data_index_hindi_tqa.npy", eng_index)
    np.save("data_indices/exemplar_idx_hindi_tqa.npy", eng_exemplar_index)


def create_bengali_tqa():
    dataset = load_dataset("truthful_qa", "generation")["validation"]
    bengali_dataset = []

    for i, item in enumerate(tqdm(dataset)):
        bengali_question = translate_with_google(item["question"], target_lang="bn")
        bengali_best_answer = translate_with_google(item["best_answer"], target_lang="bn")
        time.sleep(0.1)

        bengali_correct_answers = []
        for answer in item["correct_answers"]:
            translated = translate_with_google(answer, target_lang="bn")
            bengali_correct_answers.append(translated)
        time.sleep(0.1)

        bengali_incorrect_answers = []
        for answer in item["incorrect_answers"]:
            translated = translate_with_google(answer, target_lang="bn")
            bengali_incorrect_answers.append(translated)
        time.sleep(0.1)

        bengali_item = {
            "question": bengali_question,
            "best_answer": bengali_best_answer,
            "correct_answers": bengali_correct_answers,
            "incorrect_answers": bengali_incorrect_answers,
            "category": item["category"],
        }
        if "question_id" in item:
            bengali_item["question_id"] = item["question_id"]

        bengali_dataset.append(bengali_item)

        if (i + 1) % 100 == 0:
            with open(
                f"bengali_truthful_qa_progress_{i + 1}.json", "w", encoding="utf-8"
            ) as f:
                json.dump(bengali_dataset, f, ensure_ascii=False, indent=2)
            print(f"Saved progress: {i + 1} items translated")

    with open("bengali_truthful_qa.json", "w", encoding="utf-8") as f:
        json.dump(bengali_dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(bengali_dataset)} translated items to bengali_truthful_qa.json")

    print("\nVerifying dataset structure...")
    print(f"Original dataset fields: {list(dataset[0].keys())}")
    print(f"Bengali dataset fields: {list(bengali_dataset[0].keys())}")

    required_fields = [
        "question",
        "best_answer",
        "correct_answers",
        "incorrect_answers",
    ]
    missing_fields = [
        field for field in required_fields if field not in bengali_dataset[0]
    ]
    if missing_fields:
        print(f"Missing fields: {missing_fields}")
    else:
        print("All required fields present!")


def create_bengali_indices():
    eng_index = np.load("data_indices/data_index_tqa.npy")
    eng_exemplar_index = np.load("data_indices/exemplar_idx_tqa.npy")

    np.save("data_indices/data_index_bengali_tqa.npy", eng_index)
    np.save("data_indices/exemplar_idx_bengali_tqa.npy", eng_exemplar_index)


if __name__ == "__main__":
    # create_hindi_tqa()
    # create_hindi_indices()
    create_bengali_tqa()
    create_bengali_indices()
