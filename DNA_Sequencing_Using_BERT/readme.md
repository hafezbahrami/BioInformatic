# Fine Tunning of DNABERT
This repo contains a simple program that tries annotate the gene sequence of a Bacteria, called E_coli_K12. We will try to use a pretrained DNABERT transfomrer-encoder to label various part of the genome by "0" and "1". "0" indicates the non-coding part of the genome, while "1" shows the coding part of the genome.

## Preprocessing
preprocessing.py provides some code to read the genome files and make the test and train datasets  based on that. There are some unit test in this file that should be self-explainatory. The goal is in the main.py, we make the following calls to get the required datasets.


```
ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
preProcessObj3 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                            train_fraction=0.7, windows=[75], k_mer_val=6,
                                genome_name="ecoli")
preProcessObj3.make_datasets()
```


## Code Notes
Codes notes are in Persian, and located in below location.
![Code Notes](notes)


## Downloading the pretrained DNABERT model
The following code snippet in main.py tries to download a pretrained DNABERT and unzipp it. The unzipped files will be placed in a specific folder.
If downloading from google-drive is problemaic, the code in colab file could be helpful.

```
current_path = "./" if not debug_flag else file_dir + "/"
if fine_tune_DANABERT_using_pretrained_model:
    if not os.path.exists(current_path + "pretrained_DNA/"):
        os.makedirs(current_path + "pretrained_DNA/")
        print("A new directory is created to hold the pretrained DNAERT model.")

    # as of now the gdown only downloads in Colab, not in an script
    if not os.path.isfile(current_path + "6-new-12w-0.zip"):
        url = "https://drive.google.com/u/0/uc?id=1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC&export=download&confirm=t&uuid=574dd7fc-207b-43c4-b502-ab6a52549838&at=ALgDtswq3dLLBv3bezvOuM8dlJG-:1679328206346"
        gdown.download(url, quiet=False)

    with zipfile.ZipFile(current_path + "6-new-12w-0.zip", "r") as zip_ref:
        zip_ref.extractall(current_path + "pretrained_DNA/")
```

## Cloning DNABERT
cloning the DNABERT first:

```
git clone https://github.com/jerryji1993/DNABERT.git
```
Then make he necessary changes in versions, by calling "minor_version_changes_in_DNABERT()" method in helpere.py.
Afterward, we can install an edditable version of DNABERT in our local env.

after installing DNABERT locally, we should install the following packages in our local env.
```
tensorboardX
tensorboard
scikit-learn >= 0.22.2
seqeval
pyahocorasick
scipy
statsmodels
biopython
pandas
pybedtools
sentencepiece==0.1.99
```
