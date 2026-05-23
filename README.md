# TSV

0 - Hallucination
1 - Factually Correct



# Procedure

## Data Preparation
- First, the TruthfulQA dataset is formatted as follows:
```json
[{
    "question": string,
    "best_answer": string,
    "correct_answers": [string, string, string],
    "incorrect_answers": [string, string, string],
    "category": string
}]
```
- Then translate it to other languages like Hindi and Bengali for multilingual data

## Generation Process
- Use any LLM from the list to generate the most likely answers for each question using beam search
- This creates responses that may contain hallucinations or be factually correct

## Ground Truth (GT) Generation
- Calculate BLEURT scores for each QA pair to determine factual correctness
- BLEURT scores help identify which responses are hallucinations (0) vs factually correct (1)
- Higher BLEURT scores indicate better alignment with correct answers

## Training Process
- Train the TSV (Truthful Steering Vector) model for hallucination detection using a two-stage pipeline
- The framework minimizes human annotation costs by utilizing two types of datasets

### Data Preparation
- **Unlabeled Dataset ($\mathcal{D}_U$):** Large quantities of input prompts and LLM generations collected "in the wild" from user interactions
  - These naturally contain a mixture of both truthful (1) and hallucinated (0) content
- **Labeled Exemplar Set ($\mathcal{D}_E$):** Very small, manually annotated set of prompt-response pairs with ground-truth labels
  - Contains as few as 32 examples to keep annotation costs low
  - Labels indicate whether responses are "truthful" (1) or "hallucinated" (0)

### The Steering Mechanism (Applying TSV)
- **The Vector:** TSV is defined as a single trainable vector $v$ that can be plugged into pre-trained LLMs
- **Application during Inference:** Given a sequence of tokens, TSV is added to the intermediate layer's latent states: $h^{(l)} \leftarrow h^{(l)} + \lambda v$
  - $\lambda$ acts as a hyperparameter controlling the strength of the steering
- **Propagation:** This single vector is shared across all token positions and affects the embeddings in all subsequent layers through the LLM's non-linear transformations, ultimately reshaping the final layer's representation

### Stage 1: Initial Training Phase
- **Objective:** Establish a clear decision boundary by learning TSV to separate embeddings into two distinct classes: truthful (1) and hallucinated (0)
- **Distribution Modeling:** 
  - Treat the last-token embeddings at the final layer as a hyperspherical distribution with unit norm
  - Use von Mises-Fisher distribution to characterize this, encouraging truthful and hallucinated data to form distinct clusters
- **Loss Function:** 
  - Perform Maximum Likelihood Estimation (MLE) on the exemplar set
  - Minimize negative log-likelihood to force embeddings within each class to cluster tightly around their respective class centroids (prototypes)
- **Prototype Updating:** Class prototypes ($\mu_c$) are efficiently updated using an exponential moving average

### Stage 2: Augmented Training Phase
- **Pseudo-Labeling via Optimal Transport (OT):**
  - Use Optimal Transport-based algorithm to assign "truthful" (1) or "hallucinated" (0) labels to unlabeled data
  - Align unlabeled embeddings with established class prototypes by minimizing transport costs
  - Respect naturally imbalanced class proportions of LLM generations
  - Solve efficiently using the Sinkhorn algorithm
- **Confident Data Selection:**
  - Measure predictive uncertainty using cross-entropy
  - Select only the most "confident" pseudo-labeled samples from unlabeled dataset ($\mathcal{D}_S$)
  - Filter data to prevent noise from incorrect pseudo-labels
- **Exemplar Set Augmentation:** 
  - Combine original exemplar set with confident pseudo-labeled samples: $\mathcal{D}_E \leftarrow \mathcal{D}_E \cup \mathcal{D}_S$
- **Retraining:** Repeat learning process from Stage 1 on augmented dataset until convergence, finalizing the TSV for inference

### Model Architecture
- Uses the LLM's hidden states as input features
- Single trainable steering vector $v$ that shapes the latent space
- Prototypes (class centroids) provide stable reference points for separation
- Von Mises-Fisher distribution modeling for hyperspherical embedding clustering


## Final Output
- The trained TSV model can detect hallucinations in new LLM responses
- Uses the steering vector to identify patterns associated with hallucinations
- Supports detection but not steering of LLM outputs

---








