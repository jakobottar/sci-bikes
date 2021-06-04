import os
import csv
import argparse
import shutil

## get info about file structure from user
parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='data', help='Data Root Folder')
parser.add_argument('-V' '--verbose', dest='verbose', action='store_true')
parser.set_defaults(verbose=False)

FLAGS, unparsed = parser.parse_known_args()
root = FLAGS.root
verb = FLAGS.verbose

## create 'flat' folder in root folder to hold moved data
currDir = os.getcwd()
flat_dir = os.path.join(currDir, root, 'flat')
 
try:
    os.mkdir(flat_dir)
except FileNotFoundError:
    print('Root folder not found, exiting...')
    exit
except FileExistsError:
    if verb: print('Folder already exists, continuing...')

if verb: print(f"Making {root}.csv file...")
writer = csv.writer(open(f'{root}/flat/{root}.csv', 'w')) # will this have issues since I'm not using os.path.join?
writer.writerow(['filename', 'class'])

classes = os.listdir(root) # get a list of all subfolders of the root
classes.remove('flat')

for class_id in range(len(classes)):
    class_name = classes[class_id]
    if verb: print(f"Working on class {class_name}...")
    for file in os.listdir(os.path.join(root, class_name)):
        if file.endswith(".csv") or file.endswith(".txt"): # make sure to ignore csv and txt file
            pass

        writer.writerow([file, class_id]) # add current file to csv

        curr = os.path.join(root, class_name, file)
        targ = os.path.join(flat_dir, file)
        shutil.copy(curr, targ) # copy file to 'flat' dir