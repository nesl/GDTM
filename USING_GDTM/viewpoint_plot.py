import numpy as np
import cv2
import os
from skimage import metrics
from tqdm import tqdm
import matplotlib.pyplot as plt

metadata = []
with open("dataset_metadata.csv", "r") as fp:
    metarows = fp.readlines()
    for i in range(1,len(metarows)): # skip row 1
        temp = metarows[i].strip().split(",")
        if temp[4] == '1' and (temp[1]=="r" or temp[1]=="g"): 
        # select good illunimation + one car
            metadata.append([eval(temp[0]), eval(temp[3])])
metadata = np.array(metadata)
# for each in metadata:
#     print(each[0])

fig, axs = plt.subplots(nrows=6, ncols=3, figsize=(85,110))

data_to_look_at = [26, 39, 41, 45] # put the viewpoints you would like to visually inspect
for i in range(len(data_to_look_at)):
    v1 = metadata[data_to_look_at[i],0]
    for node1 in range(3):
        curr_node1 = "node" + str(node1+1)
        rootdir = "PATH_TO_DATASET"
        # Example folder structure
        # ---
        # ─── PATH_TO_DATASET/
        #     ├── data15
        #     │   ├── node1/
        #     │   ├── node2/
        #     |   ├── node3/
        #     |   ├── metadata.json
        #     │   └── mocap.hdf5
        #     ├── data16
        #     └── ...  
        rootfolder = os.path.join(rootdir+str(v1),curr_node1)
        filelist = os.listdir(rootfolder)
        fname1 = ""
        for each in filelist:
            if "realsense_rgb" in each: # use realsense_depth if using low lighing conditions
                fname1 = os.path.join(rootfolder, each)
                break

        vidcap = cv2.VideoCapture(fname1)
        success,image1 = vidcap.read()
        count = 0
        success = True
        while success:
            success,image1 = vidcap.read()
            count += 1
            if count>500:
                break
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        ax = axs[i][node1]
        ax.imshow(image1)
plt.show()

