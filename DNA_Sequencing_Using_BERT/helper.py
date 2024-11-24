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


          
