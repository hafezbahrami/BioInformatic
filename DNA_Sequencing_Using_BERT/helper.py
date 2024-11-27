import os
from os import path
import gdown
import zipfile
import re
import preprocessing

debug_flag = preprocessing.debug_flag
fine_tune_DANABERT_using_pretrained_model = True

file_dir = path.dirname(path.abspath(__file__))
current_path = "./" if not debug_flag else file_dir + "/"

def load_pretrained_dnabert_model():
    """This method loads a pretrained DNABERT model, for a specific XX-kmer and window-length."""       
    if fine_tune_DANABERT_using_pretrained_model:
        if not os.path.exists(current_path + "pretrained_DNA/"):
            os.makedirs(current_path + "pretrained_DNA/")
            print("A new directory is created to hold the pretrained DNAERT model.")

        # as of now the gdown only downloads in Colab, not in an script
        if not os.path.isfile(current_path + "6-new-12w-0.zip"):
            url = "https://drive.google.com/u/0/uc?id=1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC&export=download&confirm=t&uuid=574dd7fc-207b-43c4-b502-ab6a52549838&at=ALgDtswq3dLLBv3bezvOuM8dlJG-:1679328206346"
            gdown.download(url, quiet=False, output=current_path+"6-new-12w-0.zip")

        with zipfile.ZipFile(current_path + "6-new-12w-0.zip", "r") as zip_ref:
            zip_ref.extractall(current_path + "pretrained_DNA/")

#importing the regex module
import re
#defining the replace method
def replace(file_path, text, subs, flags=0):
   #open the file
   with open(file_path, "r+") as file:
       #read the file contents
       file_contents = file.read()
       text_pattern = re.compile(re.escape(text), flags)
       file_contents = text_pattern.sub(subs, file_contents)
       file.seek(0)
       file.truncate()
       file.write(file_contents)


def minor_version_changes_in_DNABERT():
    """Using RegEx to replace some of fixed versions in DNABERT, so we can get it working."""
    file_path = current_path + "DNABERT/setup.py"
    text="tokenizers == 0.5.0"
    subs="tokenizers"
    #calling the replace method
    replace(file_path, text, subs)

    file_path = current_path + "DNABERT/examples/run_finetune.py"
    text="from tqdm import tqdm, trange"
    subs="from tqdm.notebook import tqdm, trange"
    #calling the replace method
    replace(file_path, text, subs)

    file_path = current_path + "DNABERT/examples/requirements.txt"
    text="sentencepiece==0.1.91"
    subs="sentencepiece==0.1.99" # it is the latest version as of now 
    #calling the replace method
    replace(file_path, text, subs)


def _find_largest_numbered_folder(folder_path):
    # Regular expression to match folder names that end with a number
    pattern = re.compile(r'(\d+)$')

    largest_number = -1
    largest_folder = None

    # Iterate over all items in the directory
    for folder_name in os.listdir(folder_path):
        # Check if the item is a folder
        full_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(full_path):
            # Use regex to search for a number at the end of the folder name
            match = pattern.search(folder_name)
            if match:
                folder_number = int(match.group(0))  # Extract the number
                # Update if we find a larger number
                if folder_number > largest_number:
                    largest_number = folder_number
                    largest_folder = folder_name
    return folder_path + largest_folder + "/"

def setting_env_variables_for_DNABERT(kmer_val, window_size, load_nodel_from_chk_points=False):
    """Setting some enviroment variable"""
    os.environ['KMER'] = str(kmer_val)
    k = os.getenv("KMER")
    os.environ["TOKENIZER_NAME"] = f"dna{k}"

    os.environ['MODEL_PATH'] = current_path + f"pretrained_DNA/{k}-new-12w-0/"
    if load_nodel_from_chk_points:
        folder_look_up = current_path + f"output/{k}/"
        largest_chk_point_folder = _find_largest_numbered_folder(folder_look_up)
        if largest_chk_point_folder:
            os.environ['MODEL_PATH'] = largest_chk_point_folder
    
    os.environ['DATA_PATH'] = current_path + f"ecoli_data/{k}/method1/{window_size}/"
    os.environ['OUTPUT_PATH'] = current_path + f"output/{k}/"
    os.environ['PREDICTION_PATH'] = current_path + f"prediction/{k}/"

    print("\n*** printing env variable set for DNABERT...")
    print(f"k_mer length:                       {os.environ.get('KMER')}")
    print(f"Tokenizer name:                     {os.environ.get('TOKENIZER_NAME')}")
    print(f"Pre-trained DNA_BERT model path:    {os.environ.get('MODEL_PATH')}")
    print(f"Data path:                          {os.environ.get('DATA_PATH')}")
    print(f"Output path:                        {os.environ.get('OUTPUT_PATH')}")
    print(f"Predictionpath:                     {os.environ.get('PREDICTION_PATH')} \n")


    output_dir = os.getenv("OUTPUT_PATH")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pred_dir = os.getenv("PREDICTION_PATH")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)              
