import os
import random
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from tqdm import tqdm
import numpy as np
import typer
from transformers import AutoTokenizer, AutoModelForCausalLM
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from llm_layers import add_tsv_layers
from utils import Args
from train_utils import train_model, test_model


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_instruction(dataset_name: str):
    if dataset_name == "hindi_tqa":
        return "प्रश्न का उत्तर संक्षेप में कुछ वाक्यों में दें। प्रश्न: {} उत्तर:"
    else:
        return "Answer the question concisely within a few sentences. Question: {} Answer:"


def clean_generated_text(decoded: str, question: str, model_name: str) -> str:
    """Clean up generated text to remove repetitions and format issues."""
    # Remove any remaining instruction text
    instruction_patterns = [
        "Answer the question concisely within a few sentences.",
        "The answer to the question",
        "How to Write a Concise Statement",
        "You are an AI assistant",
    ]

    for pattern in instruction_patterns:
        if pattern in decoded:
            decoded = decoded.split(pattern)[0]

    # Stop at first question pattern
    if "Q:" in decoded:
        decoded = decoded.split("Q:")[0]

    # Stop at multiple choice options
    for option in ["A:", "B:", "C:", "D:"]:
        if option in decoded:
            decoded = decoded.split(option)[0]

    # Remove trailing newlines and spaces
    decoded = decoded.strip()

    # If empty after cleaning, return minimal response
    if not decoded:
        return "I cannot answer this question."

    # Ensure it ends with proper punctuation
    if not decoded.endswith((".", "!", "?")):
        decoded += "."

    return decoded


def generate_answers(
    model_name: str,
    model_name_or_path: str,
    dataset: Dataset,
    dataset_name: str,
    num_gene: int,
    most_likely: bool,
    dir_name: str,
):
    tokenizer = AutoTokenizer.from_pretrained(
        "bharatgenai/Param-1" if model_name == "param-1-2.9b" else model_name_or_path,
        trust_remote_code=False,
        attn_implementation="eager" if model_name == "param-1-2.9b" else "sdpa",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="eager" if model_name == "param-1-2.9b" else "sdpa",
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    begin_index = 0
    end_index = len(dataset)

    if not os.path.exists(f"{dir_name}/{dataset_name}_hal_det/"):
        os.mkdir(f"{dir_name}/{dataset_name}_hal_det/")
    if not os.path.exists(f"{dir_name}/{dataset_name}_hal_det/answers"):
        os.mkdir(f"{dir_name}/{dataset_name}_hal_det/answers")

    for i in range(begin_index, end_index):
        answers = [None] * num_gene
        question = dataset[i]["question"]
        instruction = (
            "प्रश्न का उत्तर संक्षेप में कुछ वाक्यों में दें। प्रश्न: {} उत्तर:"
            if dataset_name == "hindi_tqa"
            else "Answer the question concisely within a few sentences. Question: {} Aanswer:"
        )

        if model_name in ["sarvam-1", "qwen-2.5-3b", "bharatgpt-3b"]:
            conversation = [
                {
                    "role": "user",
                    "content": instruction.format(question),
                }
            ]
            prompt = tokenizer.apply_chat_template(
                conversation=conversation,
                return_tensors="pt",
                add_generation_prompt=True,
            ).cuda()
        else:
            prompt = tokenizer(
                instruction.format(question),
                return_tensors="pt",
            ).input_ids.cuda()

        with torch.no_grad():
            for gen_iter in range(num_gene):
                attention_mask = torch.where(
                    prompt == tokenizer.pad_token_id, 0, 1
                ).long()

                if most_likely:
                    generated = model.generate(
                        prompt,
                        attention_mask=attention_mask,
                        num_beams=5,
                        num_return_sequences=1,
                        do_sample=False,
                        max_new_tokens=128,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                else:
                    generated = model.generate(
                        prompt,
                        attention_mask=attention_mask,
                        do_sample=True,
                        num_return_sequences=1,
                        num_beams=1,
                        max_new_tokens=128,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                decoded = tokenizer.decode(
                    generated[0, prompt.shape[-1] :], skip_special_tokens=True
                ).strip()

                # Clean up the decoded text
                # decoded = clean_generated_text(decoded, question, model_name)

                del generated, attention_mask
                answers[gen_iter] = decoded

        del prompt
        torch.cuda.empty_cache()

        print("sample: ", i)
        if most_likely:
            info = "most_likely_"
        else:
            info = "batch_generations_"

        print(decoded)

        np.save(
            f"{dir_name}/{dataset_name}_hal_det/answers/{info}hal_det_{model_name}_{dataset_name}_answers_index_{i}.npy",
            answers,
        )


def generate_ground_truth(
    dataset: Dataset,
    dataset_name: str,
    model_name: str,
    most_likely: bool,
    dir_name: str,
):
    model = BleurtForSequenceClassification.from_pretrained(
        "lucadiliello/BLEURT-20"
    ).cuda()
    tokenizer = BleurtTokenizer.from_pretrained("lucadiliello/BLEURT-20")
    model.eval()

    print(f"Using tokenizer: {type(tokenizer).__name__}")

    gts = np.zeros(0)
    length = len(dataset)

    for i in range(length):
        if dataset_name in ("tqa", "hindi_tqa"):
            best_answer = dataset[i]["best_answer"]
            correct_answers = dataset[i]["correct_answers"]
            all_answers = [best_answer] + correct_answers
            # question = dataset[i]["question"]
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
        with torch.no_grad():
            for anw in range(len(all_answers)):
                if len(valid_predictions) == 1:
                    prediction_text = str(valid_predictions[0]).strip()
                    if not prediction_text:
                        continue
                    inputs = tokenizer(
                        [prediction_text],
                        [str(all_answers[anw])],
                        padding="longest",
                        return_tensors="pt",
                    )
                else:
                    inputs = tokenizer(
                        valid_predictions,
                        [all_answers[anw]] * len(valid_predictions),
                        padding="longest",
                        return_tensors="pt",
                    )
                for key in list(inputs.keys()):
                    inputs[key] = inputs[key].cuda()
                res = np.asarray(model(**inputs).logits.flatten().tolist())
                all_results[anw] = res
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        token="",
        attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
    )

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
            if args.dataset_name in ("tqa", "hindi_tqa")
            else None
        )

        answers = np.load(
            f"{args.dir_name}/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
        )

        for anw in answers:
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

    if args.dataset_name in ("tqa", "hindi_tqa", "triviaqa"):
        thres_gt = 0.5
    else:
        thres_gt = 0.2

    gt_label = np.asarray(gts > thres_gt, dtype=np.int32)

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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        token="",
        attn_implementation="eager" if args.model_name == "param-1-2.9b" else "sdpa",
    )

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
            if args.dataset_name in ("tqa", "hindi_tqa")
            else None
        )

        answers = np.load(
            f"{args.dir_name}/{args.dataset_name}_hal_det/answers/most_likely_hal_det_{args.model_name}_{args.dataset_name}_answers_index_{i}.npy"
        )

        for anw in answers:
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

    if args.dataset_name in ("tqa", "hindi_tqa", "triviaqa"):
        thres_gt = 0.5
    else:
        thres_gt = 0.2

    gt_label = np.asarray(gts > thres_gt, dtype=np.int32)
    index = np.load(f"data_indices/data_index_{args.dataset_name}.npy")
    wild_q_indices = index[: int(args.wild_ratio * length)]

    gt_label_test = []
    test_prompts = []
    test_qa_dicts = []

    for i in range(length):
        if i not in wild_q_indices:
            gt_label_test.extend(gt_label[i : i + 1])
            test_prompts.extend(prompts[i : i + 1])
            test_qa_dicts.extend(qa_dicts[i : i + 1])

    gt_label_test = np.asarray(gt_label_test)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    for param in model.parameters():
        param.requires_grad = False

    # Load complete checkpoint
    checkpoint = torch.load(f"{args.dir_name}/tsv_checkpoint.pt")
    centroids = checkpoint["centroids"]

    # Reconstruct TSV layers with saved parameters
    tsv = nn.ParameterList(
        [
            nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
            for _ in range(num_layers)
        ]
    )
    tsv.load_state_dict(checkpoint["tsv"])
    tsv.to(device)

    add_tsv_layers(model, tsv, [checkpoint["lam"]], args)

    test_model(
        model,
        tokenizer,
        centroids,
        test_prompts,
        gt_label_test,
        test_qa_dicts,
        device,
        args.batch_size,
        args.str_layer,
        args.dir_name,
        args.dataset_name,
        True,
    )


HF_NAMES = {
    # "llama3.2-3B": "meta-llama/Llama-3.2-3B",
    # "llama3.1-8B": "meta-llama/Meta-Llama-3.1-8B",
    # "qwen2.5-7B": "Qwen/Qwen2.5-7B",
    # "param-1": "bharatgenai/Param-1",
    # "param-1-i": "bharatgenai/Param-1-2.9B-Instruct",
    # "sarvam-1": "sarvamai/sarvam-1",
    "bharatgpt-3b": "/mnt/storage/deeksha/models/CoRover/BharatGPT-3B-Indic",
    "qwen-3-4b": "/mnt/storage/deeksha/models/Qwen/Qwen3-4B",
    "qwen-3.5-2b": "/mnt/storage/deeksha/models/Qwen/Qwen3.5-2B",
    "qwen-2.5-3b": "/mnt/storage/deeksha/models/Qwen/Qwen2.5-3B-Instruct",
    "gemma-3-4b-it": "/mnt/storage/deeksha/models/google/gemma-3-4b-it",
    "gemma-4-e2b-it": "/mnt/storage/deeksha/models/google/gemma-4-E2B-it",
    "param-1-2.9b": "/mnt/storage/deeksha/models/bharatgenai/Param-1-2.9B-Instruct",
    "sarvam-1": "/mnt/storage/deeksha/models/sarvamai/sarvam-1/",
    "llama-3.2-3b": "/mnt/storage/deeksha/models/meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2-1b": "/mnt/storage/deeksha/models/meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b": "/mnt/storage/deeksha/models/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "param-1-7b": "/mnt/storage/deeksha/models/bharatgenai/Param-1-7B",
    "krutrim-2": "/mnt/storage/deeksha/models/krutrim-ai-labs/Krutrim-2-instruct",
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
    )
    os.makedirs(args.dir_name, exist_ok=True)

    if dataset_name == "tqa":
        dataset = load_dataset("truthful_qa", "generation")["validation"]
    elif dataset_name == "hindi_tqa":
        dataset = Dataset.from_json("hindi_truthful_qa.json")
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
    else:
        raise ValueError("Invalid dataset name")

    dataset = dataset.add_column("idx", np.arange(len(dataset)))

    if gene:
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
