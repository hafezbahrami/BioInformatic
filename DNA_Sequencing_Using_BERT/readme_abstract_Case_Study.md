# 1 Using DNABERT for DNA Sequencing in E-Coli K-12 Bacteria
The genome of E. coli K-12 is a well-studied bacterial genome, consisting of a single circular chromosome with approximately 4.6 million base pairs. This strain of Escherichia coli is commonly used as a model organism in molecular biology research. Of its genome, around 88% is composed of coding sequences, which are regions that encode proteins, reflecting the efficiency of bacterial genomes. The remaining 12% of the genome is non-coding, consisting of regulatory regions, promoters, and small RNAs that play key roles in gene regulation and other cellular functions. This compact and efficient genome has been extensively studied for insights into gene expression, regulation, and bacterial physiology.

DNABERT can be used for DNA-sequencing which is translated here as for binary classification of the E. coli K-12 genome. To do so, the genome sequence must be tokenized into k-mers, typically 3 to 6 base pairs long. DNABERT, which is pre-trained on nucleotide sequences, uses a similar architecture to BERT but is fine-tuned for tasks involving DNA. For binary classification, DNABERT will be fine-tuned by preparing a labeled dataset of genome sequences, where each sequence is assigned a label (e.g., coding or non-coding). The model can then be trained to predict these labels based on the sequence inputs. After training, DNABERT can classify new sequences as belonging to one of the two categories with high accuracy, leveraging its deep understanding of nucleotide relationships within the genome.

# 2 Genome data and its preprocessing
A publicly accessible data on E-coli genome in raw format is avilable at [here](https://regulondb.ccg.unam.mx/menu/download/datasets/index.jsp).

For preprocessing data to make it ready for DNABERT the following steps are needed:

- Obtaining the raw genome sequence: The E. coli K-12 genome is typically stored in FASTA format or as a plain sequence file. It will be a long string of nucleotide bases (A, T, G, C).

- Generating k-mers: DNABERT requires the input to be tokenized into k-mers, which are overlapping substrings of nucleotides. For example, for 6-mers, the sequence ATGCGTCA will be split as ATGCGT, TGCGTC, GCGTCA, and so on. Write a script to slide a window of size k (usually 3 to 6) across the genome sequence, moving one nucleotide at a time to create overlapping k-mers.

- Labeling the data: As a binary classification is performed (e.g., coding vs. non-coding), It's needed to label each k-mer with the appropriate category. This can be done by mapping each k-mer to its corresponding genomic region (coding or non-coding) based on known annotations of the E. coli K-12 genome.


# 3 Overal goal of the project

As noted above, to model DNA sequencing for binary classification of the *E. coli* K-12 genome using **DNABERT**, the process starts with the preprocessing of the genomic data into **k-mers**. First, extract the *E. coli* genome sequence from a FASTA file. Then, divide the sequence into overlapping k-mers (subsequences of length \( k \), usually 3-6), where each k-mer is labeled as **coding** or **non-coding** based on known genome annotations. For instance, with a 6-mer approach, the sequence `ATGCGTCA` is split into overlapping k-mers: `ATGCGT`, `TGCGTC`, `GCGTCA`, etc. These labeled k-mers serve as the training data. DNABERT is pre-trained on nucleotide sequences using a transformer architecture and can be fine-tuned for this specific task by using the labeled k-mers as inputs to predict whether each sequence belongs to a coding (label = 1) or non-coding (label = 0) region. Fine-tuning involves optimizing the model to minimize classification errors by adjusting weights using gradient-based learning methods.

The model’s performance can be evaluated using standard metrics like **accuracy**, **precision**, **recall**, and **F1 score**. The accuracy is calculated as the percentage of correctly classified sequences, while precision and recall help assess the model’s ability to correctly predict coding sequences (true positives) while minimizing false positives. These metrics can be mathematically represented as:
- **Accuracy**: $$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$
- **Precision**: $$ \text{Precision} = \frac{TP}{TP + FP} $$
- **Recall** (Sensitivity): $$ \text{Recall} = \frac{TP}{TP + FN} $$
- **F1 Score**: $$ F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} $$
Where \( TP \) is true positives, \( TN \) is true negatives, \( FP \) is false positives, and \( FN \) is false negatives. Additionally, the **confusion matrix** can be presented to illustrate the classification results, showing how well the model differentiates between coding and non-coding sequences. This will help in understanding the classification behavior and potential areas for improvement.

