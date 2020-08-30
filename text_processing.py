import re
import argparse
from fastai.basics import get_files, progress_bar
from multiprocessing import Pool
import os

SAVE_TO_PATH = ""

def process_function(path_to_file):
    with open(path_to_file, 'r') as f:
        lines = f.read()
    if lines and lines[0] != ' ': lines = ' ' + lines
    lines = re.sub(r'(?=[^ ])([\W])([\w])', r'\g<1> \g<2>', lines)
    lines = re.sub(r"([\w])(?=[^ ])([\W])", r"\g<1> \g<2>", lines)
    lines = re.sub(r'([\d])([\d])', r"\g<1> \g<2>", lines)
    lines = re.sub(r'([\d])([\d])', r"\g<1> \g<2>", lines)
    lines = re.sub('(.|\s)\\1\\1+', r'\1'*3, lines)
    path = os.path.join(SAVE_TO_PATH, os.path.split(path_to_file)[1])
    with open(path, 'w') as handle:
        handle.write(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=None, type=str, 
                        required=True, help="Input folder")
    parser.add_argument("-e", "--extension", default=".txt", type=str, 
                        required=False, help="File extension.")
    parser.add_argument("-o", "--output", default=None, type=str, 
                        required=True, help="Output folder.")
    args = parser.parse_args()
    
    #process_function(args.input, args.output)
    txts = get_files(args.input, args.extension)
    global SAVE_TO_PATH
    SAVE_TO_PATH = args.output
    # Create dataset folder if not exists.
    if not os.path.exists(SAVE_TO_PATH):
        os.makedirs(SAVE_TO_PATH)

    # Process files.
    for _ in progress_bar(Pool(64).imap_unordered(process_function, txts), len(txts)):
        pass

if __name__ == "__main__":
  main()
