import os  # Import the os module
import shutil

parent = 'dataset'
for directory in next(os.walk(parent))[1]:
    try:
        shutil.rmtree(f"{parent}/{directory}")
        print(f"{parent}/{directory}")
    except OSError as err:
        print(err)

##directory = r"dataset"
##for (root,dirs,files) in os.walk('dataset.', topdown=True):
##    print (root)
##    print (dirs)
##    print (files)
##    print ('--------------------------------')

##os.chdir(directory)  # Change directory to your folder
##
### Loop through everything in folder in current working directory
##for item in os.listdir(directory):
##    if os.path.isdir(item):
##        print(item)
##        #os.remove(item)

