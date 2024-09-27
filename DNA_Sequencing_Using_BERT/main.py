import os
from os import path
import subprocess
import numpy as np

import preprocessing
import evaluation
import helper



kmer_val = 6
window_size = 75
train_fraction = 0.7
fine_tune_DANABERT_using_pretrained_model = True
debug_flag = preprocessing.debug_flag

file_dir = path.dirname(path.abspath(__file__))

def train():
    # (1) creating the datasets (write the train.tsv and dev.tsv in local disk)
    ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
    preProcessObj4 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                                train_fraction=train_fraction, windows=[window_size], k_mer_val=kmer_val,
                                    genome_name="ecoli")
    k_mer_seq_train_X_and_Y_lab_dict, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj4.make_datasets()

    # (2) Load a pretrained-DNA-BERT model
    helper.load_pretrained_dnabert_model()
    if not path.isdir(file_dir + "/DNABERT"):
        raise Exception("DNABERT must be git-cloned locally. Please read the readme.md. After cloning, the function below should be run to make the required changes. DNABERT must then be installed (as editable package).")
    helper.minor_version_changes_in_DNABERT()
    helper.setting_env_variables_for_DNABERT(kmer_val)

    # (3) TRAIN: Running fin-tunning in DNABERT: run the main() function in the DNABERT package
    TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME")
    MODEL_PATH = os.environ.get('MODEL_PATH')
    DATA_PATH = os.environ.get('DATA_PATH')
    OUTPUT_PATH = os.environ.get('OUTPUT_PATH' )
    PREDICTION_PATH = os.environ.get('PREDICTION_PATH')

    fineTuneFileNameAndLocation = file_dir + "/DNABERT/examples/run_finetune.py"
    resultTrain = subprocess.check_output(["python", fineTuneFileNameAndLocation, 
                                                        "--data_dir", DATA_PATH,                                           
                                                        "--model_type", "dna",
                                                        "--n_process", str(8),  
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--task_name", "dnaprom",  # dnaprom --> refers to DNA Promoter
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--predict_dir", PREDICTION_PATH,
                                                        "--max_seq_length", str(window_size), 
                                                        "--do_train", 
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--learning_rate", str(1e-6),
                                                        "--weight_decay", str(0.01),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--num_train_epochs", str(3.0),
                                                        "--warmup_percent", str(0.06),
                                                        "--logging_steps", str(100),
                                                        "--save_steps", str(60000),
                                                        "--overwrite_output",])
    print(resultTrain)

    # # (4) PREDICT: Running fin-tunning in prediction mode in DNABERT: run the main() function in the DNABERT package
    resultPredict = subprocess.check_output(["python", fineTuneFileNameAndLocation, 
                                                        "--model_type", "dna",
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--task_name", "dnaprom",
                                                        "--do_predict", 
                                                        "--data_dir", DATA_PATH,
                                                        "--max_seq_length", str(window_size), 
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--learning_rate", str(1e-6),
                                                        "--num_train_epochs", str(3.0),
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--predict_dir", PREDICTION_PATH,
                                                        "--logging_steps", str(100),
                                                        "--save_steps", str(60000),
                                                        "--warmup_percent", str(0.06),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--overwrite_output",
                                                        "--weight_decay", str(0.01),
                                                        "--n_process", str(8)])
    print(resultPredict)    
    
    # (5) VISUALIZATION: Running fin-tunning in visualization mode in DNABERT: run the main() function in the DNABERT package
    # After running the following snippet, we should be able to see some *.npy file in the prediction folder
    resultVisualization = subprocess.check_output(["python", fineTuneFileNameAndLocation, 
                                                        "--model_type", "dna",
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--task_name", "dnaprom",
                                                        "--do_visualize",
                                                        "--visualize_data_dir", DATA_PATH,
                                                        "--visualize_models", str(kmer_val),
                                                        "--data_dir", DATA_PATH,
                                                        "--max_seq_length", str(window_size), 
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--learning_rate", str(1e-6),
                                                        "--num_train_epochs", str(3.0),
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--predict_dir", PREDICTION_PATH,
                                                        "--logging_steps", str(100),
                                                        "--save_steps", str(60000),
                                                        "--warmup_percent", str(0.06),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--overwrite_output",
                                                        "--weight_decay", str(0.01),
                                                        "--n_process", str(8)])   
    print(resultVisualization)


    # (6) POSTPROCESSING-EVALUATION:
    if path.isdir(file_dir + f"/prediction/{kmer_val}/"):
        file_loc = file_dir + f"/prediction/{kmer_val}/pred_results.npy"
    else:
        file_loc = file_dir + f"/pred_results.npy"

    # # making up the test labels. There should be a better way to create this
    # gt_labels_test_length = 1392496
    # genome_label_test = [np.random.binomial(n=1, p=0.5, size=1).item() for _ in range(gt_labels_test_length)]
    if kmer_val != 6 or window_size != 75:
        raise Exception("Below code is temporariy and when kmer=6 and window_size=75")
    genome_label_test = []
    for line in k_mer_seq_test_X_and_Y_lab_dict["train_6_labels_75"]:
        for _ in range(window_size):
            genome_label_test.append(int(line[-2])) # This part of the code should be compatible to what we have in "predict_label_from_prob" method
    gt_labels_test = np.array(genome_label_test)

    evaluation.evaluate(datapath=file_loc, losspath=file_loc, seq_len=window_size,
            gt_labels=gt_labels_test, img_name="my_image_name", threshold=0.5)    

if __name__ == "__main__":
    train()

