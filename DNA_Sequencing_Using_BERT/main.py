import os
from os import path
import gdown
import preprocessing

fine_tune_DANABERT_using_pretrained_model = True
debug_flag = preprocessing.debug_flag

file_dir = path.dirname(path.abspath(__file__))

# # creating the datasets
# ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
# preProcessObj3 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
#                             train_fraction=0.7, windows=[75], k_mer_val=6,
#                                 genome_name="ecoli")
# preProcessObj3.make_datasets()


        
current_path = "./" if not debug_flag else file_dir + "/"
if fine_tune_DANABERT_using_pretrained_model:
    if not os.path.exists(current_path + "pretrained_DNA/"):
        os.makedirs(current_path + "pretrained_DNA/")
        print("A new directory is created to hold the pretrained DNAERT model.")

    url = "https://drive.google.com/u/0/uc?id=1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC&export=download&confirm=t&uuid=574dd7fc-207b-43c4-b502-ab6a52549838&at=ALgDtswq3dLLBv3bezvOuM8dlJG-:1679328206346"

    gdown.download(url, quiet=False)

    # !unzip -q 6-new-12w-0.zip -d "./pretrained_DNA/"
    # !rm 6-new-12w-0.zip