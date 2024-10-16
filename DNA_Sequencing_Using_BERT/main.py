import os
from os import path
import subprocess
import numpy as np

import preprocessing
import evaluation
import helper



file_dir = path.dirname(path.abspath(__file__))

def train():
    kmer_val = 6
    window_size = 75
    train_fraction = 0.7
    fine_tune_DANABERT_using_pretrained_model = True
    debug_flag = preprocessing.debug_flag
    delete_cash_files = True

    reduced_version_of_data = False                                                      # For ddebug purpose, we want to use smaller dataset
    genome_special_direction = "forward"                                                # "none", "forward", "reverse"

    save_steps = 60000                                                                  # 60000
    logging_steps = 1000                                                                # 1000

    num_train_epochs = 10                                                               # 10
    lr = 1.0e-8
    # ---------------------------------------------------------------------------------------------
    # (0) remove all cach files
    def delete_all_cash_files_recursively(directory):
        # Walk through the directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(directory):
            #print(f"Searching folder: {foldername}")  # Debug: Shows current folder
            for filename in filenames:
                # Check if the file starts with "cash"
                if filename.startswith("cach"):
                    file_path = os.path.join(foldername, filename)  # Full path to the file
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")  # Debug: Shows deleted file
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")  # Debug: Shows errors

            # Recursively call the function on subfolders
            for subfolder in subfolders:
                if subfolder != "pretrained_DNA": # exclude this folder since the pretrained model is hold here
                    subfolder_path = os.path.join(foldername, subfolder)
                    delete_all_cash_files_recursively(subfolder_path)

            break

    if delete_cash_files:
        print(f"\n\nDeleting all cach files from previous runs!")
        delete_all_cash_files_recursively(file_dir)

    # (1) creating the datasets (write the train.tsv and dev.tsv in local disk)
    if reduced_version_of_data:
        ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3_REDUCED.txt", gt_dir="./Gene_sequence_REDUCED.txt")
        num_train_epochs = 1
        logging_steps = 1
    else:
        ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
    preProcessObj4 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                                                    train_fraction=train_fraction, windows=[window_size], k_mer_val=kmer_val,
                                                    genome_name="ecoli", genome_special_direction=genome_special_direction)
    _, k_mer_seq_test_X_and_Y_lab_dict = preProcessObj4.make_datasets()

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
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--predict_dir", PREDICTION_PATH,

                                                        "--model_type", "dna",
                                                        "--task_name", "dnaprom",  # dnaprom --> refers to DNA Promoter
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--max_seq_length", str(window_size),

                                                        "--do_train", 
                                                        "--num_train_epochs", str(num_train_epochs),

                                                        "--n_process", str(8),  
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--learning_rate", str(lr),
                                                        "--weight_decay", str(0.01),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--warmup_percent", str(0.06),
                                                        "--logging_steps", str(logging_steps),
                                                        "--save_steps", str(save_steps),
                                                        "--overwrite_output",])
    print(resultTrain)

    # # (4) PREDICT: Running fin-tunning in prediction mode in DNABERT: run the main() function in the DNABERT package
    resultPredict = subprocess.check_output(["python", fineTuneFileNameAndLocation,
                                                        "--data_dir", DATA_PATH,
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--predict_dir", PREDICTION_PATH,

                                                        "--model_type", "dna",
                                                        "--task_name", "dnaprom",  # dnaprom --> refers to DNA Promoter
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--max_seq_length", str(window_size),

                                                        "--do_predict", 
                                                        "--num_train_epochs", str(3.0),

                                                        "--n_process", str(8),  
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--learning_rate", str(1e-6),
                                                        "--weight_decay", str(0.01),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--warmup_percent", str(0.06),
                                                        "--logging_steps", str(100),
                                                        "--save_steps", str(60000),
                                                        "--overwrite_output",])
    print(resultPredict)    
    
    # (5) VISUALIZATION: Running fin-tunning in visualization mode in DNABERT: run the main() function in the DNABERT package
    # After running the following snippet, we should be able to see some *.npy file in the prediction folder
    resultVisualization = subprocess.check_output(["python", fineTuneFileNameAndLocation,
                                                        "--data_dir", DATA_PATH,
                                                        "--model_name_or_path", MODEL_PATH,
                                                        "--output_dir", OUTPUT_PATH,
                                                        "--predict_dir", PREDICTION_PATH,

                                                        "--model_type", "dna",
                                                        "--task_name", "dnaprom",  # dnaprom --> refers to DNA Promoter
                                                        "--tokenizer_name", TOKENIZER_NAME,
                                                        "--max_seq_length", str(window_size),

                                                        "--do_visualize",
                                                        "--visualize_data_dir", DATA_PATH,
                                                        "--visualize_models", str(kmer_val), 
                                                        "--num_train_epochs", str(3.0),

                                                        "--n_process", str(8),  
                                                        "--per_gpu_train_batch_size", str(16),
                                                        "--per_gpu_eval_batch_size", str(16),
                                                        "--learning_rate", str(1e-6),
                                                        "--weight_decay", str(0.01),
                                                        "--hidden_dropout_prob", str(0.1),
                                                        "--warmup_percent", str(0.06),
                                                        "--logging_steps", str(100),
                                                        "--save_steps", str(60000),
                                                        "--overwrite_output",])  
    print(resultVisualization)


    # (6) POSTPROCESSING-EVALUATION:
    if path.isdir(file_dir + f"/prediction/{kmer_val}/"):
        pred_file_loc = file_dir + f"/prediction/{kmer_val}/pred_results.npy"
    else:
        pred_file_loc = file_dir + f"/pred_results.npy"
    
    if path.exists(OUTPUT_PATH + "loss.txt"):
        loss_file_loc = OUTPUT_PATH + "loss.txt"

    # # making up the test labels. There should be a better way to create this
    # gt_labels_test_length = 1392496
    # genome_label_test = [np.random.binomial(n=1, p=0.5, size=1).item() for _ in range(gt_labels_test_length)]
    if kmer_val != 6 or window_size != 75:
        raise Exception("Below code is temporariy and when kmer=6 and window_size=75")
    genome_label_test = []
    for line in k_mer_seq_test_X_and_Y_lab_dict["test_6_labels_75"]:
        for _ in range(window_size):
            if line[-2].isdigit():
                genome_label_test.append(int(line[-2])) # This part of the code should be compatible to what we have in "predict_label_from_prob" method
    gt_labels_test = np.array(genome_label_test)

    evaluation.evaluate(datapath=pred_file_loc, losspath=loss_file_loc, seq_len=window_size,
            gt_labels=gt_labels_test, img_name="all_res_together.png", threshold=0.5)    

if __name__ == "__main__":
    train()

