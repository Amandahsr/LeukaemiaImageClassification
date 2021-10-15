#Create new directory to store augmentation test results.
! mkdir test_augment

#Copy 3 images each from ALL and Hem classes into test_augment directory.
! find /content/datasets/C-NMC_Leukemia/training_data/fold_0/all -type f | head -3 | xargs cp -t /content/test_augment
! find /content/datasets/C-NMC_Leukemia/training_data/fold_0/hem -type f | head -3 | xargs cp -t /content/test_augment

#Initialise input/output directory for testing.
ip_dir = "/content/test_augment/"
op_dir = "/content/test_augment/augmented/"

#Run data augmentation code. 
main(rounds=5, ip_dir=ip_dir, op_dir=op_dir)

# Visualise images created from augmentation test.
fig = plt.figure(figsize=(12, 4))
row = 2
column = 6

fig.add_subplot(row, column, 1)
plt.imshow(cv2.imread("/content/test_augment/UID_28_34_5_all.bmp"))
plt.axis('off')
plt.title("ALL_Org")

for i in range(5):
  fig.add_subplot(row, column, i+2)
  plt.imshow(cv2.imread(f"/content/test_augment/augmented/UID_28_34_5_all_{i}.bmp"))
  plt.axis('off')
  plt.title(f"ALL_{i}")

fig.add_subplot(row, column, 7)
plt.imshow(cv2.imread("/content/test_augment/UID_H11_4_1_hem.bmp"))
plt.axis('off')
plt.title("HEM_Org")

for j in range(5):
  fig.add_subplot(row, column, j+8)
  plt.imshow(cv2.imread(f"/content/test_augment/augmented/UID_H11_4_1_hem_{j}.bmp"))
  plt.axis('off')
  plt.title(f"HEM_{j}")

filenames = os.listdir("/content/datasets/C-NMC_Leukemia/training_data/fold_2/hem")[1:10]

for bmp in filenames:
  files.download("/content/datasets/C-NMC_Leukemia/training_data/fold_2/hem/"+bmp)
