#Custom CellsDataset class for dataloader to process images.
class CellsDataset(Dataset):

    def __init__(self, label_file, img_dir):
        try:
            self.img_labels = pd.read_csv(label_file) #Label csv files.
        except:
            self.img_labels = label_file #Label df files.

        self.img_dir = img_dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 2]

        return image, label
