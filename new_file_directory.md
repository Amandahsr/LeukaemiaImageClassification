### New file directory:
    Data_main:
    | - held_out_test
    |   | - test_images
    |   | - test_labels.csv

    | - images

    | - labels.csv
    | - train_labels.csv

test_images:
- .bmp files of new test set.

images:
- .bmp files of new train set.

csv files:
- labels.csv: labels of all .bmp files.
- train_labels.csv: labels for new train set (subset of labels.csv).
- test_labels.csv: labels for new test set (subset of labels.csv).
- csv format:

    | Patient_no      | Patient_ID | labels |
    | ----------- | ----------- | ---|
    | N      | UID_P_N_C_{all/hem}.bmp       | 1/0 |
    | ...      | ...       | ... |
