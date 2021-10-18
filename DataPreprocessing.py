import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------
# REFORMAT DIRECTORY FOR TRAINING #
# ---------------------------------------------
# as per google colab file structure
INPUT_DIR = "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data Subset/"
OUTPUT_DIR = "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/"

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR + "images/")
    os.mkdir(OUTPUT_DIR + "held_out_test/")
    os.mkdir(OUTPUT_DIR + "held_out_test/test_images")

# (1) Process validation_data
valid_filepath = INPUT_DIR + "validation_data/C-NMC_test_prelim_phase_data/"
valid_df = pd.read_csv(INPUT_DIR + "validation_data/C-NMC_test_prelim_phase_data_labels.csv")

valid_l = []
for idx, row in valid_df.iterrows():
    id = row["Patient_ID"]
    id = id.split('.')[0] + ".png" # convert to png image
    patient_no = id.split('_')[1]
    filename = row["new_names"]
    labels = row["labels"]
    # rename validation_data image filename: ({int}.bmp) to (UID_{id}_{P}_{N}_{C}_{'all/hem'})
    # copy .bmp into images folder
    if os.path.exists(valid_filepath + filename):
      Image.open(valid_filepath + filename).save(OUTPUT_DIR + "images/" + id)
        # shutil.copy(valid_filepath + filename, OUTPUT_DIR + "images/" + id)
    valid_l.append([patient_no, id, labels])

valid_df = pd.DataFrame(valid_l, columns=["Patient_no", "Patient_ID", "labels"])

# (2) Process training_data
train_l = []
for dir, sub, files in os.walk(INPUT_DIR + "training_data"):
    for bmp in files:
        ip_filepath = dir +"/"+ bmp
        png = bmp.split('.')[0] + ".png"
        op_filepath = OUTPUT_DIR + "images/" + png

        # copy .bmp into images folder
        if not os.path.exists(op_filepath):
            # shutil.copy(ip_filepath, op_filepath)
            Image.open(ip_filepath).save(op_filepath)

        # append into csv file
        label = 1 if ("all" in bmp) else 0
        patient_no = bmp.split('_')[1]
        train_l.append([patient_no, png, label])

train_df = pd.DataFrame(train_l, columns=["Patient_no", "Patient_ID", "labels"])

# merge training and validation labels into csv file
merge_df = pd.concat([train_df, valid_df], axis=0)
merge_df.to_csv(OUTPUT_DIR + 'labels.csv')

# ---------------------------------------------
# PICK IMAGES FOR HELD OUT TEST SET #
# ALL: 11
# HEM: 4
# Total patients: 15
# ---------------------------------------------

# # Look at no. of images per patient by Patient_ID
# patient_id = merge_df["Patient_ID"]
# all_count = []
# hem_count = []
# for id in patient_id:
#     pt = id.split('_')[1]
#     if 'H' in pt:
#         hem_count.append(int(pt[1:]))
#     else:
#         all_count.append(int(pt))

# print("Total ALL: {} ; Total HEM: {}".format(len(all_count), len(hem_count)))

# # Visualization
# plt.hist(all_count, bins=78)
# plt.hist(hem_count, bins=48)
# plt.title('Patient_ID distribution')
# plt.xlabel("Patient_ID")
# plt.ylabel("Frequency")
# # plt.show()
# # plt.savefig("Patient_ID_distribution.png")

# # Enumerate frequencies
# import collections
# all_freq = collections.Counter(all_count)
# hem_freq = collections.Counter(hem_count)
# # print(all_freq)
# # print(hem_freq)

# pick out images from ALL (11) and HEM (4) based on Patient_ID frequencies
test_pick = ['26', '35', '44', '33', '24', '27', '25', '46', '49', '37', '31', 'H12', 'H17', 'H2', 'H15']
# eg_pick = ['45', 'H11'] # small testcase

# create new csv file for held_out_test
test_df = merge_df.loc[merge_df['Patient_no'].isin(test_pick)]
test_df.to_csv(OUTPUT_DIR + 'held_out_test/test_labels.csv')

# move test images into held_out_test folder
files = test_df["Patient_ID"].tolist()
for bmp in files:
    shutil.move(OUTPUT_DIR + 'images/' + bmp, OUTPUT_DIR + 'held_out_test/test_images/' + bmp)

# create new train data labels
training_df = pd.concat([merge_df, test_df]).drop_duplicates(keep=False)
training_df.to_csv(OUTPUT_DIR + 'train_labels.csv')

print("Train shape: {}, Test shape: {}, Merge shape: {}".format(training_df.shape, test_df.shape, merge_df.shape))
