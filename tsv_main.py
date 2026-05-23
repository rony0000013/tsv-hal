import os
import csv
import json
import random
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import typer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMultimodalLM
from llm_layers import add_tsv_layers
from utils import Args
from train_utils import train_model, test_model

TEMPLATE_LLMS = ["sarvam-1", "qwen-2.5-3b", "bharatgpt-3b", "gemma-3-4b-it", "olmo-3-7b", "nanda-10b"]

# Configuration for torch.compile optimization
USE_TORCH_COMPILE = True  # Set to False to disable torch.compile for debugging

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_instruction(dataset_name: str, language: str = None):
    if dataset_name == "hindi_tqa":
        return "प्रश्न का उत्तर संक्षेप में कुछ (1-2) वाक्यों में दें। प्रश्न: {} उत्तर:"
    elif dataset_name == "ben_tqa":
        return "প্রশ্নটির উত্তর কয়েকটি বাক্যে সংক্ষেপে দিন (1-2)। প্রশ্ন: {} উত্তর:"
    elif dataset_name == "combined_tqa":
        # Language-specific instruction for combined multilingual training
        if language == "hindi_tqa":
            return "प्रश्न का उत्तर संक्षेप में कुछ (1-2) वाक्यों में दें। प्रश्न: {} उत्तर:"
        elif language == "ben_tqa":
            return "প্রশ্নটির উত্তর কয়েকটি বাক্যে সংক্ষেপে দিন (1-2)। প্রশ্ন: {} উত্তর:"
        else:  # English (tqa) or default
            return "Answer the question concisely within a few (1-2) sentences. Question: {} Answer:"
    else:
        return "Answer the question concisely within a few (1-2) sentences. Question: {} Answer:"


def generate_answers(
    model_name: str,
    model_name_or_path: str,
    dataset: Dataset,
    dataset_name: str,
    num_gene: int,
    most_likely: bool,
    dir_name: str,
):
    # Fix rope_scaling config for Param 1 model
    if model_name == "ministral-3-3b":
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",
        )
    
    # Apply torch.compile for performance optimization
    if USE_TORCH_COMPILE:
        try:
            print("Applying torch.compile for performance optimization...")
            # Use max-autotune for best performance, fallback to default if needed
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("torch.compile applied successfully")
        except Exception as e:
            print(f"torch.compile failed, using uncompiled model: {e}")
            # Fallback to reduced optimization or no compilation
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile applied with reduce-overhead mode")
            except Exception as e2:
                print(f"torch.compile completely failed, using uncompiled model: {e2}")
    else:
        print("torch.compile disabled by configuration")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    instruction = get_instruction(dataset_name)
    begin_index = 0
    end_index = len(dataset)
    
    # Pre-create directories
    os.makedirs(f"{dir_name}/{dataset_name}_hal_det/answers", exist_ok=True)
    
    # Setup CSV file once
    info = "most_likely_" if most_likely else "batch_generations_"
    csv_path = f"{dir_name}/{dataset_name}_hal_det/{info}answer.csv"
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow(['index', 'output'])
    
    # Batch processing variables
    batch_size = 32  # Process in batches for efficiency
    csv_batch = []
    
    print(f"Generating {end_index - begin_index} answers with batch size {batch_size}")
    
    for batch_start in range(begin_index, end_index, batch_size):
        batch_end = min(batch_start + batch_size, end_index)
        current_batch_size = batch_end - batch_start
        
        # Process batch
        for i in range(batch_start, batch_end):
            answers = [None] * num_gene
            question = dataset[i]["question"]

            # Create prompt
            if model_name in TEMPLATE_LLMS:
                conversation = [{"role": "user", "content": instruction.format(question)}]
                inputs = tokenizer.apply_chat_template(
                    conversation=conversation,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                prompt = inputs.input_ids.cuda()
            else:
                prompt = tokenizer(instruction.format(question), return_tensors="pt").input_ids.cuda()

            # Generate answers
            with torch.no_grad():
                attention_mask = torch.where(prompt == tokenizer.pad_token_id, 0, 1).long()
                
                # Optimize for single generation (most common case)
                if num_gene == 1:
                    generation_kwargs = {
                        "input_ids": prompt,  # Use input_ids instead of prompt for torch.compile compatibility
                        "attention_mask": attention_mask,
                        "max_new_tokens": 128,
                        "eos_token_id": tokenizer.eos_token_id,
                        "pad_token_id": tokenizer.pad_token_id,
                        "use_cache": True,
                    }
                    
                    if most_likely:
                        generation_kwargs.update({
                            "num_beams": 3,
                            "num_return_sequences": 1,
                            "do_sample": False,
                            "repetition_penalty": 1.2,
                        })
                    else:
                        generation_kwargs.update({
                            "do_sample": True,
                            "num_return_sequences": 1,
                            "num_beams": 1,
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "repetition_penalty": 1.2,
                        })

                    generated = model.generate(**generation_kwargs)
                    decoded = tokenizer.decode(generated[0, prompt.shape[-1]:], skip_special_tokens=True).strip()
                    answers = [decoded]
                else:
                    # Handle multiple generations efficiently
                    for gen_iter in range(num_gene):
                        generation_kwargs = {
                            "input_ids": prompt,  # Use input_ids instead of prompt for torch.compile compatibility
                            "attention_mask": attention_mask,
                            "max_new_tokens": 128,
                            "eos_token_id": tokenizer.eos_token_id,
                            "pad_token_id": tokenizer.pad_token_id,
                            "use_cache": True,
                        }
                        
                        if most_likely:
                            generation_kwargs.update({
                                "num_beams": 5,
                                "num_return_sequences": 1,
                                "do_sample": False,
                            })
                        else:
                            generation_kwargs.update({
                                "do_sample": True,
                                "num_return_sequences": 1,
                                "num_beams": 1,
                            })

                        generated = model.generate(**generation_kwargs)
                        decoded = tokenizer.decode(generated[0, prompt.shape[-1]:], skip_special_tokens=True).strip()
                        answers[gen_iter] = decoded
                        del generated
                
                del attention_mask

            # Save answers immediately
            np.save(
                f"{dir_name}/{dataset_name}_hal_det/answers/{info}hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy",
                answers,
            )
            
            # Add to CSV batch
            csv_batch.append((i, decoded))
            
            del prompt
            torch.cuda.empty_cache()

        # Write CSV batch
        writer.writerows(csv_batch)
        csv_file.flush()  # Ensure data is written
        csv_batch.clear()
        
        # Progress update (less frequent)
        if (batch_end - begin_index) % 100 == 0 or batch_end == end_index:
            print(f"Processed {batch_end - begin_index}/{end_index - begin_index} samples")
    
    # Close CSV file
    csv_file.close()
    print(f"Generation complete. Results saved to {csv_path}")


def generate_ground_truth(
    dataset: Dataset,
    dataset_name: str,
    model_name: str,
    most_likely: bool,
    dir_name: str,
):
    # Use Google's BLEURT library for true BLEURT scoring
    from bleurt import score as bleurt_score
    print("Loading BLEURT scorer...")
    bleurt_scorer = bleurt_score.BleurtScorer()
    print("Using BLEURT scorer directly")

    gts = np.zeros(0)
    length = len(dataset)

    for i in range(length):
        if dataset_name in ("tqa", "hindi_tqa", "ben_tqa"):
            best_answer = dataset[i]["best_answer"]
            correct_answers = dataset[i]["correct_answers"]
            all_answers = [best_answer] + correct_answers
        elif dataset_name == "triviaqa":
            all_answers = dataset[i]["answer"]["aliases"]

        if most_likely:
            answers = np.load(
                f"{dir_name}/{dataset_name}_hal_det/answers/most_likely_hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy"
            )
        else:
            answers = np.load(
                f"{dir_name}/{dataset_name}_hal_det/answers/batch_generations_hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy"
            )

        predictions = answers
        print(
            f"Sample predictions: {predictions[:3] if hasattr(predictions, '__getitem__') else 'no indexing'}"
        )

        valid_predictions = []
        for pred in predictions:
            if pred is not None and str(pred).strip():
                valid_predictions.append(str(pred))

        print(f"Valid predictions count: {len(valid_predictions)}")
        print(f"Sample valid predictions: {valid_predictions[:3]}")

        if len(valid_predictions) == 0:
            print("Skipping due to no valid predictions")
            continue

        all_results = np.zeros((len(all_answers), len(valid_predictions)))
        
        # Use true BLEURT scoring
        for anw in range(len(all_answers)):
            reference = str(all_answers[anw])
            if len(valid_predictions) == 1:
                prediction_text = str(valid_predictions[0]).strip()
                if not prediction_text:
                    continue
                scores = bleurt_scorer.score(references=[reference], candidates=[prediction_text])
            else:
                scores = bleurt_scorer.score(references=[reference] * len(valid_predictions), candidates=valid_predictions)
            all_results[anw] = np.array(scores)
        gts = np.concatenate([gts, np.max(all_results, axis=0)], 0)
        if i % 10 == 0:
            print("samples passed: ", i)

    if most_likely:
        np.save(f"{dir_name}/ml_{dataset_name}_bleurt_score.npy", gts)

    else:
        np.save(f"{dir_name}/bg_{dataset_name}_bleurt_score.npy", gts)


def train_fn(model_name_or_path: str, dataset: Dataset, args: Args):
    import os
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        "bharatgenai/Param-1" if args.model_name == "param-1-2.9b" else model_name_or_path,
        attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
        trust_remote_code=False,
    )
    
    if args.model_name == "ministral-3-3b":
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token="",
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token="",
            attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
        )
    
    # Apply torch.compile for training performance optimization
    if USE_TORCH_COMPILE:
        try:
            print("Applying torch.compile for training optimization...")
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("torch.compile applied successfully for training")
        except Exception as e:
            print(f"torch.compile failed for training, using uncompiled model: {e}")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile applied with reduce-overhead mode for training")
            except Exception as e2:
                print(f"torch.compile completely failed for training, using uncompiled model: {e2}")
    else:
        print("torch.compile disabled by configuration for training")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    prompts = []
    qa_dicts = []
    length = len(dataset)
    instruction = get_instruction(args.dataset_name)

    for i in tqdm(range(length)):
        question = dataset[i]["question"]
        best_answer = dataset[i]["best_answer"]
        category = (
            dataset[i]["category"]
            if args.dataset_name in ("tqa", "hindi_tqa", "ben_tqa")
            else None
        )
        
        # Get language-specific instruction for combined dataset
        if args.dataset_name == "combined_tqa":
            language = dataset[i]["language"]
            instruction = get_instruction(args.dataset_name, language)
        else:
            instruction = get_instruction(args.dataset_name)

        answers = np.load(
            f"{args.dir_name}/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
        )

        for anw in answers:
            if args.model_name in TEMPLATE_LLMS:
                conversation = [
                    {
                        "role": "user",
                        "content": f"{instruction.format(question)} {anw}",
                    }
                ]
                inputs = tokenizer.apply_chat_template(
                    conversation=conversation,
                    return_tensors="pt",
                    add_generation_prompt=False,
                )
                prompt = inputs.input_ids.cuda()
            else:
                prompt = tokenizer(
                    f"{instruction.format(question)} {anw}",
                    return_tensors="pt",
                ).input_ids.cuda()

            prompts.append(prompt)
            qa_dicts.append(
                {
                    "Question": question,
                    "Answer": anw,
                    "Best Answer": best_answer,
                    "Category": category,
                }
            )

    gts = np.load(f"{args.dir_name}/ml_{args.dataset_name}_bleurt_score.npy")
    length = len(dataset)

    if args.dataset_name in ("tqa", "hindi_tqa", "ben_tqa", "triviaqa", "combined_tqa"):
        thres_gt = 0.5
    else:
        thres_gt = 0.2

    gt_label = np.asarray(gts > thres_gt, dtype=np.int32)

    # Load data indices from correct location for combined dataset
    if args.dataset_name == "combined_tqa":
        index = np.load(f"{args.dir_name}/data_index_{args.dataset_name}.npy")
        exemplar_index = np.load(f"{args.dir_name}/exemplar_idx_{args.dataset_name}.npy")
    else:
        index = np.load(f"data_indices/data_index_{args.dataset_name}.npy")
        exemplar_index = np.load(f"data_indices/exemplar_idx_{args.dataset_name}.npy")
    wild_q_indices = index[: int(args.wild_ratio * length)]
    wild_q_indices1 = wild_q_indices[: len(wild_q_indices) - 100]

    gt_label_test = []
    gt_label_wild = []
    gt_label_exemplar = []

    test_prompts = []
    train_prompts = []
    exemplar_prompts = []

    test_qa_dicts = []
    train_qa_dicts = []
    exemplar_qa_dicts = []

    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i : i + 1])
            test_prompts.extend(prompts[i : i + 1])
            test_qa_dicts.extend(qa_dicts[i : i + 1])
        elif i in exemplar_index:
            gt_label_exemplar.extend(gt_label[i : i + 1])
            exemplar_prompts.extend(prompts[i : i + 1])
            train_qa_dicts.extend(qa_dicts[i : i + 1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i : i + 1])
            train_prompts.extend(prompts[i : i + 1])
            exemplar_qa_dicts.extend(qa_dicts[i : i + 1])

    gt_label_test = np.asarray(gt_label_test)
    gt_label_exemplar = np.asarray(gt_label_exemplar)
    gt_label_wild = np.asarray(gt_label_wild)

    labels = (gt_label_test, gt_label_wild, gt_label_exemplar)
    prompts = (test_prompts, train_prompts, exemplar_prompts)
    qa_dicts = (test_qa_dicts, train_qa_dicts, exemplar_qa_dicts)

    # Handle multimodal models like Ministral-3-3B
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
    else:
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

    for param in model.parameters():
        param.requires_grad = False

    tsv = nn.ParameterList(
        [
            nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            for _ in range(num_layers)
        ]
    )
    tsv.to(device)

    add_tsv_layers(model, tsv, [args.lam], args)

    optimizer = torch.optim.AdamW(tsv.parameters(), lr=args.lr)

    train_model(
        model,
        tokenizer,
        optimizer,
        tsv,
        device,
        prompts,
        labels,
        qa_dicts,
        args=args,
    )


def test_fn(model_name_or_path: str, dataset: Dataset, args: Args):
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        "bharatgenai/Param-1" if args.model_name == "param-1-2.9b" else model_name_or_path,
        attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
        trust_remote_code=False,
    )
    
    if args.model_name == "ministral-3-3b":
        model = AutoModelForMultimodalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token="",
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            token="",
            attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
        )
    
    # Apply torch.compile for testing performance optimization
    if USE_TORCH_COMPILE:
        try:
            print("Applying torch.compile for testing optimization...")
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
            print("torch.compile applied successfully for testing")
        except Exception as e:
            print(f"torch.compile failed for testing, using uncompiled model: {e}")
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("torch.compile applied with reduce-overhead mode for testing")
            except Exception as e2:
                print(f"torch.compile completely failed for testing, using uncompiled model: {e2}")
    else:
        print("torch.compile disabled by configuration for testing")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    prompts = []
    qa_dicts = []
    length = len(dataset)
    instruction = get_instruction(args.dataset_name)

    for i in tqdm(range(length)):
        question = dataset[i]["question"]
        best_answer = dataset[i]["best_answer"]
        category = (
            dataset[i]["category"]
            if args.dataset_name in ("tqa", "hindi_tqa", "ben_tqa")
            else None
        )
        
        # Get language-specific instruction for combined dataset
        if args.dataset_name == "combined_tqa":
            language = dataset[i]["language"]
            instruction = get_instruction(args.dataset_name, language)
        else:
            instruction = get_instruction(args.dataset_name)

        answers = np.load(
            f"{args.dir_name}/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
        )

        for anw in answers:
            if args.model_name in TEMPLATE_LLMS:
                conversation = [
                    {
                        "role": "user",
                        "content": f"{instruction.format(question)} {anw}",
                    }
                ]
                inputs = tokenizer.apply_chat_template(
                    conversation=conversation,
                    return_tensors="pt",
                    add_generation_prompt=False,
                )
                prompt = inputs.input_ids.cuda()
            else:
                prompt = tokenizer(
                    f"{instruction.format(question)} {anw}",
                    return_tensors="pt",
                ).input_ids.cuda()

            prompts.append(prompt)
            qa_dicts.append(
                {
                    "Question": question,
                    "Answer": anw,
                    "Best Answer": best_answer,
                    "Category": category,
                }
            )

    gts = np.load(f"{args.dir_name}/ml_{args.dataset_name}_bleurt_score.npy")
    length = len(dataset)

    if args.dataset_name in ("tqa", "hindi_tqa", "ben_tqa", "triviaqa", "combined_tqa"):
        thres_gt = 0.5
    else:
        thres_gt = 0.2

    gt_label = np.asarray(gts > thres_gt, dtype=np.int32)
    
    # Load data indices from correct location for combined dataset
    if args.dataset_name == "combined_tqa":
        index = np.load(f"{args.dir_name}/data_index_{args.dataset_name}.npy")
        exemplar_index = np.load(f"{args.dir_name}/exemplar_idx_{args.dataset_name}.npy")
    else:
        index = np.load(f"data_indices/data_index_{args.dataset_name}.npy")
        exemplar_index = np.load(f"data_indices/exemplar_idx_{args.dataset_name}.npy")
    wild_q_indices = index[: int(args.wild_ratio * length)]
    wild_q_indices1 = wild_q_indices[: len(wild_q_indices) - 100]

    gt_label_test = []
    gt_label_wild = []
    gt_label_exemplar = []

    test_prompts = []
    train_prompts = []
    exemplar_prompts = []

    test_qa_dicts = []
    train_qa_dicts = []
    exemplar_qa_dicts = []

    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i : i + 1])
            test_prompts.extend(prompts[i : i + 1])
            test_qa_dicts.extend(qa_dicts[i : i + 1])
        elif i in exemplar_index:
            gt_label_exemplar.extend(gt_label[i : i + 1])
            exemplar_prompts.extend(prompts[i : i + 1])
            train_qa_dicts.extend(qa_dicts[i : i + 1])
        elif i in wild_q_indices1:
            gt_label_wild.extend(gt_label[i : i + 1])
            train_prompts.extend(prompts[i : i + 1])
            exemplar_qa_dicts.extend(qa_dicts[i : i + 1])

    gt_label_test = np.asarray(gt_label_test)
    gt_label_exemplar = np.asarray(gt_label_exemplar)
    gt_label_wild = np.asarray(gt_label_wild)

    labels = (gt_label_test, gt_label_wild, gt_label_exemplar)
    prompts = (test_prompts, train_prompts, exemplar_prompts)
    qa_dicts = (test_qa_dicts, train_qa_dicts, exemplar_qa_dicts)

    # Handle multimodal models like Ministral-3-3B
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
    else:
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size

    for param in model.parameters():
        param.requires_grad = False

    # Load centroids and checkpoint (cross-language support)
    if args.external_centroids_path and args.external_checkpoint_path:
        print(f"Loading external centroids from: {args.external_centroids_path}")
        print(f"Loading external checkpoint from: {args.external_checkpoint_path}")
        print(f"Source language: {args.source_language}")
        print(f"Target language: {args.dataset_name}")
        
        # Load external centroids
        centroids = torch.load(args.external_centroids_path)
        
        # Load external checkpoint
        checkpoint = torch.load(args.external_checkpoint_path)
        
        # Validate compatibility
        if centroids.shape[1] != hidden_size:
            print(f"WARNING: Centroids dimension {centroids.shape[1]} != model hidden size {hidden_size}")
            print("This may indicate incompatible model architectures.")
            
        # Use checkpoint args for TSV configuration
        checkpoint_args = checkpoint["args"]
        lam = checkpoint_args.get("lam", args.lam)
        component = checkpoint_args.get("component", args.component)
        str_layer = checkpoint_args.get("str_layer", args.str_layer)
        
        print(f"Using TSV config from source: lam={lam}, component={component}, str_layer={str_layer}")
        
    else:
        # Load complete checkpoint from current directory
        checkpoint = torch.load(f"{args.dir_name}/tsv_checkpoint.pt")
        centroids = checkpoint["centroids"]
        lam = checkpoint["lam"]
        component = args.component
        str_layer = args.str_layer

    # Reconstruct TSV layers with saved parameters
    tsv = nn.ParameterList(
        [
            nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            for _ in range(num_layers)
        ]
    )
    tsv.load_state_dict(checkpoint["tsv"])
    tsv.to(device)

    # Create a temporary args object for add_tsv_layers
    args_dict = args.__dict__.copy()
    args_dict.update({
        'lam': lam,
        'component': component,
        'str_layer': str_layer
    })
    temp_args = Args(**args_dict)
    
    add_tsv_layers(model, tsv, [lam], temp_args)

    val_predictions, val_labels_combined, _ = test_model(
        model,
        centroids,
        test_prompts,
        gt_label_test,
        test_qa_dicts,
        device,
        args.batch_size,
        str_layer,
        args.dir_name,
        args.dataset_name,
        False,
    )

    val_predictions_np = val_predictions.cpu().numpy()
    val_labels_np = val_labels_combined.cpu().numpy()     
    
    auroc = roc_auc_score(val_labels_np, val_predictions_np)
    print(f"Test AUROC: {auroc:.4f}")
    print(f"Test samples: {len(val_labels_np)}")
    print(f"Label distribution: {np.bincount(val_labels_np.astype(int))}")
    print(f"Prediction range: [{val_predictions_np.min():.3f}, {val_predictions_np.max():.3f}]")


def combine_multilingual_datasets(model_name: str, str_layer: int):
    """
    Combine multilingual datasets into a single directory for training.
    Copies pregenerated answers and GT scores from individual language directories.
    """
    languages = ["tqa", "hindi_tqa", "ben_tqa"]
    combined_dir = f"TSV_{model_name}_combined_tqa_{str_layer}"
    
    print(f"Creating combined multilingual dataset in {combined_dir}")
    
    # Create combined directory structure
    os.makedirs(f"{combined_dir}/combined_tqa_hal_det/answers", exist_ok=True)
    
    # Combine all datasets
    combined_dataset = []
    combined_qa_dicts = []
    combined_gts = []
    
    # Language-agnostic instruction (use English as base)
    instruction = "Answer the question concisely within a few (1-2) sentences. Question: {} Answer:"
    
    for lang_idx, lang in enumerate(languages):
        print(f"Processing {lang}...")
        
        # Load dataset
        if lang == "tqa":
            dataset = load_dataset("truthful_qa", "generation")["validation"]
        elif lang == "hindi_tqa":
            dataset = Dataset.from_json("hindi_truthful_qa.json")
        elif lang == "ben_tqa":
            dataset = Dataset.from_json("bengali_truthful_qa.json")
        
        # Load ground truth scores
        lang_dir = f"TSV_{model_name}_{lang}_{str_layer}"
        gt_path = f"{lang_dir}/ml_{lang}_bleurt_score.npy"
        
        if not os.path.exists(gt_path):
            print(f"WARNING: GT scores not found at {gt_path}")
            continue
            
        gts = np.load(gt_path)
        
        # Process each sample
        for i in range(len(dataset)):
            question = dataset[i]["question"]
            best_answer = dataset[i]["best_answer"]
            category = dataset[i]["category"] if lang in ("tqa", "hindi_tqa", "ben_tqa") else None
            
            # Copy pregenerated answers to combined directory
            src_answer_path = f"{lang_dir}/{lang}_hal_det/answers/most_likely_hal_det_{model_name}_{lang}_answers_index_{i}.npy"
            dst_answer_path = f"{combined_dir}/combined_tqa_hal_det/answers/most_likely_hal_det_{model_name}_combined_tqa_answers_index_{lang_idx * 817 + i}.npy"
            
            if os.path.exists(src_answer_path):
                answers = np.load(src_answer_path)
                np.save(dst_answer_path, answers)
            else:
                print(f"WARNING: Answer file not found at {src_answer_path}")
                continue
            
            # Add to combined dataset
            combined_dataset.append({
                "question": question,
                "best_answer": best_answer,
                "category": category,
                "language": lang,
                "original_index": i
            })
            
            combined_qa_dicts.append({
                "Question": question,
                "Answer": answers[0] if len(answers) > 0 else "",
                "Best Answer": best_answer,
                "Category": category,
                "Language": lang
            })
            
            combined_gts.append(gts[i])
    
    # Save combined GT scores
    combined_gts = np.array(combined_gts)
    np.save(f"{combined_dir}/ml_combined_tqa_bleurt_score.npy", combined_gts)
    
    # Save combined dataset as JSON
    with open(f"{combined_dir}/combined_tqa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(combined_dataset, f, ensure_ascii=False, indent=2)
    
    # Create combined data indices
    total_samples = len(combined_dataset)
    print(f"Combined dataset created with {total_samples} samples")
    
    # Generate data indices for combined dataset (similar to individual languages)
    np.random.seed(42)
    indices = np.random.permutation(total_samples)
    
    # Split indices: 75% for wild (training), 25% for test
    wild_ratio = 0.75
    wild_count = int(total_samples * wild_ratio)
    
    wild_indices = indices[:wild_count]
    test_indices = indices[wild_count:]
    
    # Create exemplar indices from wild indices (32 per language = 96 total)
    exemplar_count_per_lang = 32
    exemplar_count = exemplar_count_per_lang * 3  # 32 * 3 languages = 96 total
    
    # Ensure we don't exceed available wild indices
    exemplar_count = min(exemplar_count, len(wild_indices))
    exemplar_indices = np.random.choice(wild_indices, exemplar_count, replace=False)
    
    # Save combined data indices
    np.save(f"{combined_dir}/data_index_combined_tqa.npy", indices)
    np.save(f"{combined_dir}/exemplar_idx_combined_tqa.npy", exemplar_indices)
    
    print(f"Combined dataset preparation complete!")
    print(f"- Total samples: {total_samples}")
    print(f"- Wild (training) samples: {len(wild_indices)}")
    print(f"- Test samples: {len(test_indices)}")
    print(f"- Exemplar samples: {len(exemplar_indices)}")
    print(f"- Directory: {combined_dir}")


HF_NAMES = {
    # "llama3.2-3B": "meta-llama/Llama-3.2-3B",
    # "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    # "qwen2.5-7B": "Qwen/Qwen2.5-7B",
    # "param-1": "bharatgenai/Param-1",
    # "param-1-i": "bharatgenai/Param-1-2.9B-Instruct",
    # "sarvam-1": "sarvamai/sarvam-1",
    "olmo-3-7b": "/mnt/storage/deeksha/models/allenai/Olmo-3-7B-Instruct",
    "nanda-10b": "/mnt/storage/deeksha/models/Llama-3-Nanda-10B-Chat",
    "granite-4-3b": "/mnt/storage/deeksha/models/ibm-granite/granite-4.0-micro",
    "airavata-8b": "/mnt/storage/deeksha/models/ai4bharat/Airavata",
    "ministral-3-3b": "/mnt/storage/deeksha/models/mistralai/Ministral-3-3B-Instruct-2512",
    "bharatgpt-3b": "/mnt/storage/deeksha/models/CoRover/BharatGPT-3B-Indic",
    "qwen-3-4b": "/mnt/storage/deeksha/models/Qwen/Qwen3-4B",
    "qwen-2.5-3b": "/mnt/storage/deeksha/models/Qwen/Qwen2.5-3B-Instruct",
    "gemma-3-4b-it": "/mnt/storage/deeksha/models/google/gemma-3-4b-it",
    "gemma-4-e2b-it": "/mnt/storage/deeksha/models/google/gemma-4-E2B-it",
    "param-1-2.9b": "/mnt/storage/deeksha/models/bharatgenai/Param-1-2.9B-Instruct",
    "sarvam-1": "/mnt/storage/deeksha/models/sarvamai/sarvam-1/",
    "llama-3.2-3b": "/mnt/storage/deeksha/models/meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b": "/mnt/storage/deeksha/models/meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "/mnt/storage/deeksha/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "param-1-7b": "/mnt/storage/deeksha/models/bharatgenai/Param-1-7B",
    "krutrim-1-7b": "/mnt/storage/deeksha/models/krutrim-ai-labs/Krutrim-1-instruct",
    "krutrim-2-12b": "/mnt/storage/deeksha/models/krutrim-ai-labs/Krutrim-2-instruct",
}


def main(
    model_name: str = typer.Option("llama-3.2-3B", help="Model name"),
    model_prefix: str = typer.Option("", help="Prefix of model name"),
    num_gene: int = typer.Option(1, help="Number of generations"),
    gene: bool = typer.Option(False, help="Gene flag", is_flag=True),
    generate_gt: bool = typer.Option(
        False, help="Generate ground truth flag", is_flag=True
    ),
    train: bool = typer.Option(False, help="Train flag", is_flag=True),
    test: bool = typer.Option(False, help="Test flag", is_flag=True),
    dataset_name: str = typer.Option("tqa", help="Dataset name"),
    most_likely: bool = typer.Option(False, help="Most likely flag", is_flag=True),
    wild_ratio: float = typer.Option(0.75, help="Wild ratio"),
    thres_gt: float = typer.Option(0.5, help="Ground truth threshold"),
    model_dir: str = typer.Option(None, help="Local directory with model data"),
    batch_size: int = typer.Option(32, help="Batch size"),
    cos_temp: float = typer.Option(0.1, help="Cosine temperature"),
    ema_decay: float = typer.Option(0.99, help="EMA decay"),
    lr: float = typer.Option(0.005, help="Learning rate"),
    str_layer: int = typer.Option(9, help="Start layer"),
    component: str = typer.Option("res", help="Component type"),
    lam: float = typer.Option(5, help="Lambda parameter"),
    init_num_epochs: int = typer.Option(20, help="Initial number of epochs"),
    aug_num_epochs: int = typer.Option(20, help="Augmented number of epochs"),
    num_exemplars: int = typer.Option(32, help="Number of exemplars"),
    num_selected_data: int = typer.Option(128, help="Number of selected data"),
    cls_dist: str = typer.Option("proxy", help="Class distribution"),
    optimizer: str = typer.Option("AdamW", help="Optimizer type"),
    num_iters_sk: int = typer.Option(3, help="Number of Sinkhorn iterations"),
    epsilon_sk: float = typer.Option(0.05, help="Sinkhorn epsilon"),
    # Cross-language arguments
    external_centroids_path: str = typer.Option(None, help="Path to external centroids file for cross-language testing"),
    external_checkpoint_path: str = typer.Option(None, help="Path to external TSV checkpoint file for cross-language testing"),
    source_language: str = typer.Option(None, help="Source language name for cross-language testing (e.g., 'hindi_tqa', 'ben_tqa')"),
    # Combined multilingual arguments
    combine: bool = typer.Option(False, help="Combine multilingual datasets flag", is_flag=True),
):
    model_name_or_path = HF_NAMES[model_prefix + model_name]

    args = Args(
        model_name=model_name,
        model_prefix=model_prefix,
        dir_name=f"TSV_{model_name}_{dataset_name}_{str_layer}",
        num_gene=num_gene,
        dataset_name=dataset_name,
        most_likely=most_likely,
        wild_ratio=wild_ratio,
        thres_gt=thres_gt,
        model_dir=model_dir,
        batch_size=batch_size,
        cos_temp=cos_temp,
        ema_decay=ema_decay,
        lr=lr,
        str_layer=str_layer,
        component=component,
        lam=lam,
        init_num_epochs=init_num_epochs,
        aug_num_epochs=aug_num_epochs,
        num_exemplars=num_exemplars,
        num_selected_data=num_selected_data,
        cls_dist=cls_dist,
        optimizer=optimizer,
        num_iters_sk=num_iters_sk,
        epsilon_sk=epsilon_sk,
        # Cross-language parameters
        external_centroids_path=external_centroids_path,
        external_checkpoint_path=external_checkpoint_path,
        source_language=source_language,
        # Combined multilingual parameters
        combine=combine,
    )
    os.makedirs(args.dir_name, exist_ok=True)

    if dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", "generation")["validation"]
    elif dataset_name == "hindi_tqa":
        dataset = Dataset.from_json("hindi_truthful_qa.json")
    elif dataset_name == "ben_tqa":
        dataset = Dataset.from_json("bengali_truthful_qa.json")
    elif dataset_name == "triviaqa":
        dataset = load_dataset("trivia_qa", "rc.nocontext", split="validation")
        id_mem = set()

        def remove_dups(batch):
            if batch["question_id"][0] in id_mem:
                return {_: [] for _ in batch.keys()}
            id_mem.add(batch["question_id"][0])
            return batch

        dataset = dataset.map(
            remove_dups, batch_size=1, batched=True, load_from_cache_file=False
        )
    elif dataset_name == "sciq":
        dataset = load_dataset("allenai/sciq", split="validation")
    elif dataset_name == "nq_open":
        dataset = load_dataset("google-research-datasets/nq_open", split="validation")
    elif dataset_name == "combined_tqa":
        # Load combined multilingual dataset
        combined_dir = f"TSV_{model_name}_combined_tqa_{str_layer}"
        dataset_path = f"{combined_dir}/combined_tqa_dataset.json"
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Combined dataset not found at {dataset_path}. Please run with --combine flag first.")
        
        dataset = Dataset.from_json(dataset_path)
    else:
        raise ValueError("Invalid dataset name")

    dataset = dataset.add_column("idx", np.arange(len(dataset)))

    if combine:
        combine_multilingual_datasets(model_name, str_layer)
        return
    elif gene:
        generate_answers(
            model_name,
            model_name_or_path,
            dataset,
            dataset_name,
            num_gene,
            most_likely,
            args.dir_name,
        )
    elif generate_gt:
        generate_ground_truth(
            dataset, dataset_name, model_name, most_likely, args.dir_name
        )
    elif train:
        train_fn(model_name_or_path, dataset, args)
    elif test:
        test_fn(model_name_or_path, dataset, args)
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    seed_everything(42)
    typer.run(main)
