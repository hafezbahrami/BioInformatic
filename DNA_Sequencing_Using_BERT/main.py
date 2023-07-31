import os
from os import path
import subprocess


import preprocessing
from helper import load_pretrained_dnabert_model, minor_version_changes_in_DNABERT, setting_env_variables_for_DNABERT



kmer_val = 6
window_size = 75
train_fraction = 0.7
fine_tune_DANABERT_using_pretrained_model = True
debug_flag = preprocessing.debug_flag

file_dir = path.dirname(path.abspath(__file__))

def train():
    # (1) creating the datasets
    ecoli_genome, gt_gen_seq_coor, _, _ = preprocessing._get_data(genome_seq_dir="./E_coli_K12_MG1655_U00096.3.txt", gt_dir="./Gene_sequence.txt")
    preProcessObj3 = preprocessing.PreProcessData(genome=ecoli_genome, gt_gen_seq_coor=gt_gen_seq_coor,
                                train_fraction=train_fraction, windows=[window_size], k_mer_val=kmer_val,
                                    genome_name="ecoli")
    preProcessObj3.make_datasets()

    # (2) Load a pretrained-DNA-BERT model
    load_pretrained_dnabert_model()
    if not path.isdir(file_dir + "/DNABERT"):
        raise Exception("DNABERT must be git-cloned locally. Please read the readme.md. After cloning, the function below should be run to make the required changes. DNABERT must then be installed.")
    minor_version_changes_in_DNABERT()
    setting_env_variables_for_DNABERT(kmer_val)

    # (3) TRAIN: Running fin-tunning in DNABERT: run the main() function in the DNABERT package
    TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME")
    MODEL_PATH = os.environ.get('MODEL_PATH')
    DATA_PATH = os.environ.get('DATA_PATH')
    OUTPUT_PATH = os.environ.get('OUTPUT_PATH' )
    PREDICTION_PATH = os.environ.get('PREDICTION_PATH')

    fineTuneFileNameAndLocation = file_dir + "/DNABERT/examples/run_finetune.py"
    # resultTrain = subprocess.check_output(["python", fineTuneFileNameAndLocation, 
    #                                                     "--model_type", "dna",
    #                                                     "--tokenizer_name", TOKENIZER_NAME,
    #                                                     "--model_name_or_path", MODEL_PATH,
    #                                                     "--task_name", "dnaprom",
    #                                                     "--do_train", 
    #                                                     "--data_dir", DATA_PATH,
    #                                                     "--max_seq_length", str(window_size), 
    #                                                     "--per_gpu_eval_batch_size", str(16),
    #                                                     "--per_gpu_train_batch_size", str(16),
    #                                                     "--learning_rate", str(1e-6),
    #                                                     "--num_train_epochs", str(3.0),
    #                                                     "--output_dir", OUTPUT_PATH,
    #                                                     "--predict_dir", PREDICTION_PATH,
    #                                                     "--logging_steps", str(100),
    #                                                     "--save_steps", str(60000),
    #                                                     "--warmup_percent", str(0.06),
    #                                                     "--hidden_dropout_prob", str(0.1),
    #                                                     "--overwrite_output",
    #                                                     "--weight_decay", str(0.01),
    #                                                     "--n_process", str(8)])
    # print(resultTrain)

    # # (4) PREDICT: Running fin-tunning in prediction mode in DNABERT: run the main() function in the DNABERT package
    # resultPredict = subprocess.check_output(["python", fineTuneFileNameAndLocation, 
    #                                                     "--model_type", "dna",
    #                                                     "--tokenizer_name", TOKENIZER_NAME,
    #                                                     "--model_name_or_path", MODEL_PATH,
    #                                                     "--task_name", "dnaprom",
    #                                                     "--do_predict", 
    #                                                     "--data_dir", DATA_PATH,
    #                                                     "--max_seq_length", str(window_size), 
    #                                                     "--per_gpu_eval_batch_size", str(16),
    #                                                     "--per_gpu_train_batch_size", str(16),
    #                                                     "--learning_rate", str(1e-6),
    #                                                     "--num_train_epochs", str(3.0),
    #                                                     "--output_dir", OUTPUT_PATH,
    #                                                     "--predict_dir", PREDICTION_PATH,
    #                                                     "--logging_steps", str(100),
    #                                                     "--save_steps", str(60000),
    #                                                     "--warmup_percent", str(0.06),
    #                                                     "--hidden_dropout_prob", str(0.1),
    #                                                     "--overwrite_output",
    #                                                     "--weight_decay", str(0.01),
    #                                                     "--n_process", str(8)])
    # print(resultPredict)    
    
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

if __name__ == "__main__":
    train()

