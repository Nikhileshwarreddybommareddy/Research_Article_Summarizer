# Research Article Summarizer 

## Abstract
Our project is all about making life easier for students and researchers who often find themselves swamped with long research papers. We're using Natural Language Processing (NLP) and Generative AI to create short, helpful summaries of these papers. This way, students can quickly decide whether a paper is relevant to their work without having to read the whole thing. Our approach involves two main strategies. The first is to pick out the key sentences from the original text and put them together to form a summary. The second is a bit more complex, where we actually write new sentences that capture the main ideas of the paper in a nutshell. Our aim is to make these summaries really good, even better than what's already out there.

## Methodology

### 1. Data Collection
We employed the Scisumm dataset from the ACL Anthology network, comprising over 1,000 research papers with corresponding expert-written summaries. This dataset includes:
   - Document XML files detailing each paper's structure and content.
   - Summary folders containing expert-crafted summaries.

### 2. Text Preprocessing
The preprocessing steps involve:
   - Excluding non-essential sections (e.g., above "Abstract", acknowledgements, references).
   - Utilizing spaCy for cleaning and organizing the text.
   - Converting data into a structured format for model training.

### 3. Tokenization and Train-Test Split
We used the Pegasus tokenizer for its proficiency in text summarization tasks. Our data, split into training and validation sets (90% and 10% respectively), underwent tokenization and were organized into a custom `SummaryDataset` class using PyTorch.

### 4. Model Training
We explored two primary approaches:
   - **Extractive Summarization using Page Rank Algorithm:** This method selects high-importance sentences based on the PageRank algorithm.
   - **Abstractive Summarization using Pegasus Model:** Involves both pre-trained and fine-tuned Pegasus models for generating new, coherent summaries. Additionally, we experimented with the T5 model for its text-to-text transfer capabilities.

### 5. Output and Evaluation
Comparative analysis of summaries generated from different models was conducted. Our fine-tuned Pegasus model demonstrated superior performance in summarizing a sample climate change paper, indicating its efficacy in research article summarization. The Rouge Scores for different models are tabulated as follows:

| Model                           | Rouge_1 | Rouge_2 | Rouge_L |
|---------------------------------|---------|---------|---------|
| Extractive using Page Rank      | 0.18    | 0.10    | 0.12    |
| T5 Transformer                  | 0.36    | 0.24    | 0.29    |
| Pegasus Pre-Trained             | 0.40    | 0.23    | 0.29    |
| Pegasus Fine-Tuned              | 0.58    | 0.45    | 0.50    |

### 6. Model Deployment
The fine-tuned Pegasus model was deployed on a website using Flask, offering users options for summary length (short, medium, long). Users input a research article and receive a processed summary.

### 7. Possible Improvements and Considerations
Future enhancements include benchmarking against other summarization models, iterative training with new data, and addressing ethical considerations like bias.

## Conclusion
This project aims to deliver a tool that not only summarizes text but also encapsulates the essence of complex research articles in a concise manner, addressing the challenges of language processing and domain-specific nuances.

## Reference
- [SciSummNet - Scientific Article Summarization Dataset](https://cs.stanford.edu/~myasu/projects/scisumm_net/)
- [Finetuning Pegasus model - Hugging Face Discussion](https://discuss.huggingface.co/t/fine-tuning-pegasus/1433)