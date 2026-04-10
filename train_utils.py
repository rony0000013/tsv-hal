from typing import List, Dict
from utils import (
    get_last_non_padded_token_rep,
    compute_ot_loss_cos,
    update_centroids_ema,
    update_centroids_ema_hard,
    get_ex_data,
    collate_fn,
)
import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
from transformers import PreTrainedModel, AutoTokenizer
from sklearn.metrics import roc_auc_score
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from sinkhorn_knopp import SinkhornKnopp_imb
import logging
import wandb
from utils import Args


def clean_answer(prompt: str, dataset_name: str) -> str:
    if dataset_name == "hindi_tqa":
        if "उत्तर:" in prompt:
            return prompt.split("उत्तर:")[1]
        elif "जवाब:" in prompt:
            return prompt.split("जवाब:")[1]
        else:
            return prompt
    else:
        if "A:" in prompt:
            return prompt.split("A:")[1]
        elif "Answer:" in prompt:
            return prompt.split("Answer:")[1]
        else:
            return prompt


def setup_multi_device():
    """Setup multi-device configuration with model parallelism if available"""
    num_gpus = torch.cuda.device_count()
    
    # Use single GPU to avoid conflicts with Accelerate hooks
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using single GPU setup: {device} (Avoiding Accelerate conflicts)")
    return [device], device, False, "single"


def train_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.AdamW,
    tsv: torch.nn.ParameterList,
    device: torch.device,
    prompts: (List, List, List),
    labels: (np.array, np.array, np.array),
    qa_dicts: (Dict, Dict, Dict),
    args: Args,
):
    # Setup multi-device configuration
    model_devices, data_device, use_multi_device, device_mode = setup_multi_device()
    primary_model_device = model_devices[0]
    
    # Use single device to avoid Accelerate conflicts
    # Don't move the model - let Accelerate handle device placement
    print(f"Using single device setup: Letting Accelerate manage device placement")
    
    layer_number = -1

    # dir_name = f"TSV_{args.model_name}_{args.dataset_name}/exemplar_num_{args.num_exemplars}_num_selected_data_{args.num_selected_data}/{args.component}/{args.str_layer}/{args.lam}"
    # dir_name = f"TSV_{args.model_name}_{args.dataset_name}_{args.str_layer}"
    # log_dir = dir_name
    # log_file = os.path.join(log_dir, "log.txt")

    wandb.init(
        project="tsv",
        name=f"{args.model_name}_{args.dataset_name}",
        config=args.__dict__,
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Starting training")
    logging.info(
        f"Training parameters: few_shot_size={args.num_exemplars}, num_selected_data={args.num_selected_data}, component={args.component}, str_layer={args.str_layer}"
    )

    test_prompts, train_prompts, exemplar_prompts = prompts
    test_labels, train_labels, exemplar_labels = labels
    test_qa_dicts, _, _ = qa_dicts
    batch_size = args.batch_size
    num_samples = len(train_prompts)
    num_exemplars = args.num_exemplars
    num_epochs = args.init_num_epochs

    best_test_auroc = -1

    scaler = GradScaler(device=device)

    # Get model device from Accelerate-managed model
    if hasattr(model, 'module'):
        hidden_size = model.module.config.hidden_size
        model_device = next(model.module.parameters()).device
    else:
        hidden_size = model.config.hidden_size
        model_device = next(model.parameters()).device

    # Initialize Sinkhorn algorithm
    args.num_iters_sk = 3
    args.epsilon_sk = 0.05

    ex_hallu = (num_exemplars - exemplar_labels[:num_exemplars].sum()) / num_exemplars
    ex_true = (exemplar_labels[:num_exemplars].sum()) / num_exemplars
    cls_dist = torch.tensor([ex_hallu, ex_true]).float().to(model_device)
    cls_dist = cls_dist.view(-1, 1)
    sinkhorn = SinkhornKnopp_imb(args, cls_dist)

    # Initialize Centroids - will be moved to match model output device dynamically
    centroids = torch.randn((2, hidden_size)).half()
    centroids = F.normalize(centroids, p=2, dim=1)

    exemplar_prompts_ = exemplar_prompts
    exemplar_prompts, exemplar_labels = collate_fn(exemplar_prompts, exemplar_labels)

    for epoch in range(num_epochs):
        running_loss = 0.0
        total = 0
        num_samples = num_exemplars

        # Process data in batches
        for batch_start in tqdm(
            range(0, num_samples, batch_size),
            desc=f"Epoch {epoch + 1}/{num_epochs} Batches",
            leave=False,
        ):
            batch_prompts = exemplar_prompts[batch_start : batch_start + batch_size]
            batch_labels = exemplar_labels[batch_start : batch_start + batch_size]
            attention_mask = (batch_prompts != 0).half()

            batch_prompts = batch_prompts.to(model_device)
            batch_labels = batch_labels.to(model_device)
            attention_mask = attention_mask.to(model_device)
            
            with autocast(device_type="cuda", dtype=torch.float16):
                # Use the base model to skip the massive LM head/logits calculation
                base_model = model.model if hasattr(model, "model") else model
                output = base_model(
                    batch_prompts.squeeze(1),
                    attention_mask=attention_mask.squeeze(1),
                    output_hidden_states=True,
                )

                # Index the hidden_states tuple directly instead of stacking to save GBs of VRAM
                last_layer_hidden_state = output.hidden_states[layer_number]
                # Use attention mask to ignore padding tokens, and get the last non-padded token's representation
                last_token_rep = get_last_non_padded_token_rep(
                    last_layer_hidden_state, attention_mask.squeeze(1)
                )
                # Move centroids to same device as last_token_rep
                centroids = centroids.to(last_token_rep.device)
                batch_labels_oh = torch.nn.functional.one_hot(
                    batch_labels, num_classes=-1
                )
                ot_loss, similarities = compute_ot_loss_cos(
                    last_token_rep, centroids, batch_labels_oh, batch_size, args
                )
                loss = ot_loss
                total += batch_labels.size(0)

                with torch.no_grad():
                    centroids = update_centroids_ema_hard(
                        centroids, last_token_rep, batch_labels_oh, args
                    )

            # loss.backward()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            running_loss += loss.item() * batch_labels.size(0)

        # Epoch summary
        epoch_loss = running_loss / total

        if (epoch + 1) % 1 == 0:
            test_predictions, test_labels_combined, df = test_model(
                model,
                tokenizer,
                centroids,
                test_prompts,
                test_labels,
                test_qa_dicts,
                device,
                batch_size,
                layer_number,
                args.dir_name,
                args.dataset_name,
                (epoch + 1) == num_epochs,
            )
            test_auroc = roc_auc_score(
                test_labels_combined.cpu().numpy(), test_predictions.cpu().numpy()
            )

            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc

            if (epoch + 1) == num_epochs and df is not None:
                wandb.log({"test_predictions": wandb.Table(dataframe=df)})
            wandb.log(
                {
                    "train1/loss": epoch_loss,
                    "train1/epoch": epoch,
                    "train1/test_auroc": test_auroc,
                }
            )
            # print(
            #     f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f} ,Test AUROC: {test_auroc:.4f}"
            # )
            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f} ,Test AUROC: {test_auroc:.4f}"
            )

    logging.info("SS Learning Starts")

    with torch.no_grad():
        selected_indices, selected_labels_soft = get_ex_data(
            model,
            train_prompts,
            train_labels,
            batch_size,
            centroids,
            sinkhorn,
            args.num_selected_data,
            cls_dist,
            args,
        )

        num_samples = len(selected_indices) + args.num_exemplars

    num_epochs = args.aug_num_epochs
    exemplar_label = (
        exemplar_labels.cuda() if hasattr(exemplar_labels, "cuda") else exemplar_labels
    )

    selected_prompts = [train_prompts[i] for i in selected_indices]
    selected_labels = selected_labels_soft
    # selected_qa_dicts = [qa_dicts[i] for i in selected_indices]

    augmented_prompts_train = selected_prompts + exemplar_prompts_
    exemplar_labels = torch.nn.functional.one_hot(
        exemplar_label.to(torch.int64), num_classes=2
    )
    # Move selected_labels to same device as exemplar_labels
    selected_labels = selected_labels.to(model_device)
    augmented_labels_label = torch.concat(
        (selected_labels, exemplar_labels.clone().to(model_device))
    )

    num_samples = len(augmented_prompts_train)

    with autocast(device_type="cuda", dtype=torch.float16):
        for epoch in range(num_epochs):
            running_loss = 0.0
            total = 0

            for batch_start in tqdm(
                range(0, num_samples, batch_size),
                desc=f"Epoch {epoch + 1}/{num_epochs} Batches",
                leave=False,
            ):
                batch_prompts = augmented_prompts_train[
                    batch_start : batch_start + batch_size
                ]
                batch_labels = augmented_labels_label[
                    batch_start : batch_start + batch_size
                ]

                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

                # Shape: [batch_size, max_seq_len]
                attention_mask = (batch_prompts != 0).half()

                batch_prompts = batch_prompts.to(primary_model_device)
                batch_labels = batch_labels.to(primary_model_device)
                attention_mask = attention_mask.to(primary_model_device)
                
                # Use the base model to skip the massive LM head/logits calculation
                base_model = model.model if hasattr(model, "model") else model
                output = base_model(
                    batch_prompts.squeeze(1),
                    attention_mask=attention_mask.squeeze(1),
                    output_hidden_states=True,
                )

                # Index the hidden_states tuple directly instead of stacking to save GBs of VRAM
                last_layer_hidden_state = output.hidden_states[layer_number]
                # Use attention mask to ignore padding tokens, and get the last non-padded token's representation
                # Shape: [batch_size, hidden_size]
                last_token_rep = get_last_non_padded_token_rep(
                    last_layer_hidden_state, attention_mask.squeeze(1)
                )
                # Move centroids to same device as last_token_rep
                centroids = centroids.to(last_token_rep.device)
                ot_loss, similarities = compute_ot_loss_cos(
                    last_token_rep, centroids, batch_labels, batch_size, args
                )

                loss = ot_loss

                with torch.no_grad():
                    centroids = update_centroids_ema(
                        centroids, last_token_rep, batch_labels.half(), args
                    )
                    total += batch_labels.size(0)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # Accumulate the loss
                running_loss += loss.item() * batch_labels.size(0)

            epoch_loss = running_loss / total  # Normalize loss by total samples

            with torch.no_grad():
                if epoch % 1 == 0:
                    test_predictions, test_labels_combined, df = test_model(
                        model,
                        tokenizer,
                        centroids,
                        test_prompts,
                        test_labels,
                        test_qa_dicts,
                        device,
                        batch_size,
                        layer_number,
                        args.dir_name,
                        args.dataset_name,
                        (epoch + 1) == num_epochs,
                    )
                    test_auroc = roc_auc_score(test_labels_combined, test_predictions)

            # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")
            if (epoch + 1) == num_epochs and df is not None:
                wandb.log({"test_predictions": wandb.Table(dataframe=df)})
            if test_auroc > best_test_auroc:
                best_test_auroc = test_auroc

            logging.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test AUROC: {test_auroc:.4f} "
            )
            wandb.log(
                {
                    "train2/loss": epoch_loss,
                    "train2/epoch": epoch,
                    "train2/test_auroc": test_auroc,
                }
            )

    wandb.log({
        "centroids": wandb.Histogram(centroids.cpu().numpy()),
        "centroids_mean": centroids.mean().item(),
        "centroids_std": centroids.std().item(),
        "num_centroids": centroids.shape[0],
        "final/best_auroc": best_test_auroc,
        "final/layer_number": args.str_layer,
        "final/component": args.component,
        "final/lam": args.lam
    })
    torch.save(centroids, f"{args.dir_name}/centroids.pt")
    torch.save(
        {
            "centroids": centroids,
            "tsv": tsv.state_dict(),
            "args": args.__dict__,
            "best_auroc": best_test_auroc,
            "layer_number": args.str_layer,
            "component": args.component,
            "lam": args.lam,
        },
        f"{args.dir_name}/tsv_checkpoint.pt",
    )


    wandb.finish()
    return best_test_auroc


def test_model(
    model: PreTrainedModel,
    tokenizer: AutoTokenizer,
    centroids: torch.Tensor,
    test_prompts: List,
    test_labels: np.array,
    test_qa_dicts: Dict,
    device: torch.device,
    batch_size: int,
    layer_number: int,
    dir_name: str,
    dataset_name: str,
    last_epoch=False,
) -> (torch.Tensor, torch.Tensor, pd.DataFrame):
    model.eval()
    val_predictions = []
    val_labels_combined = []

    # all_last_token_reps = []
    # all_labels = []
    rows_list = []  # Use list for efficient row collection

    num_val_samples = len(test_prompts)

    with torch.no_grad():
        with autocast(device_type="cuda", dtype=torch.float16):
            for batch_start in range(0, num_val_samples, batch_size):
                batch_prompts = test_prompts[batch_start : batch_start + batch_size]
                batch_labels = test_labels[batch_start : batch_start + batch_size]
                batch_prompts, batch_labels = collate_fn(batch_prompts, batch_labels)

                # Get model device for test function
                if hasattr(model, 'module'):
                    test_model_device = next(model.module.parameters()).device
                else:
                    test_model_device = next(model.parameters()).device
                    
                attention_mask = (batch_prompts != 0).half().to(test_model_device)
                batch_prompts = batch_prompts.to(test_model_device)
                batch_labels = batch_labels.to(test_model_device)
                
                # Forward pass - use the base model to skip the expensive LM head/logits calculation
                base_model = model.model if hasattr(model, "model") else model
                output = base_model(
                    batch_prompts.squeeze(1),
                    attention_mask=attention_mask.squeeze(1),
                    output_hidden_states=True,
                )
                # Index the hidden_states tuple directly instead of stacking to save GBs of VRAM
                last_layer_hidden_state = output.hidden_states[layer_number]
                last_token_rep = get_last_non_padded_token_rep(
                    last_layer_hidden_state, attention_mask.squeeze(1)
                )

                # all_last_token_reps.append(
                #     F.normalize(last_token_rep, p=2, dim=-1).detach().cpu().numpy()
                # )
                # all_labels.append(batch_labels.cpu().numpy())

                last_token_rep = F.normalize(last_token_rep, p=2, dim=-1)
                # Move centroids to same device as last_token_rep
                centroids = centroids.to(last_token_rep.device)
                centroids = F.normalize(centroids, p=2, dim=-1)

                with autocast(device_type="cuda", dtype=torch.float16):
                    # Shape: [256, 2]
                    similarities = torch.matmul(last_token_rep, centroids.T)

                similarity_scores = torch.softmax(similarities / 0.1, dim=-1)
                similarity_scores = similarity_scores[:, 1]
                val_predictions.append(similarity_scores.cpu())
                val_labels_combined.append(batch_labels.cpu())

                # Log input text and predictions for this batch
                if tokenizer is not None and last_epoch is True:
                    for i in range(batch_prompts.shape[0]):
                        # Decode input text (remove padding tokens)
                        input_ids = batch_prompts[
                            i, 0
                        ]  # Remove the extra dimension from collate_fn
                        attention_mask_i = attention_mask[i]

                        # Find the actual length (non-padding tokens)
                        actual_length = int(attention_mask_i.sum().item())
                        truncated_ids = (
                            input_ids[:actual_length].cpu().numpy()
                        )  # Convert to numpy array

                        # Decode text
                        input_text = tokenizer.decode(
                            truncated_ids.tolist(), skip_special_tokens=True
                        )

                        # Get prediction and true label
                        pred_score = similarity_scores[i].item()
                        true_label = batch_labels[i].item()
                        pred_class = 1 if pred_score >= 0.5 else 0

                        new_row = {
                            "idx": batch_start + i,
                            "Question": test_qa_dicts[batch_start + i]["Question"],
                            "Best Answer": test_qa_dicts[batch_start + i][
                                "Best Answer"
                            ],
                            "Answer": clean_answer(input_text, dataset_name),
                            "Truth Label": true_label,
                            "Predicted Label": pred_class,
                        }
                        rows_list.append(new_row)
                        # test_logs.append(f'Input: "{input_text}"')
                        # test_logs.append(
                        #     f"Prediction Score: {pred_score:.4f}, Predicted Class: {pred_class}, True Label: {true_label}"
                        # )

    val_predictions = torch.cat(val_predictions)
    val_labels_combined = torch.cat(val_labels_combined)

    if last_epoch:
        df = pd.DataFrame(rows_list)
        df.to_csv(f"{dir_name}/predictions.csv", index=False)
        print("Saved predictions csv")
        return val_predictions, val_labels_combined, df
    return val_predictions, val_labels_combined, None
