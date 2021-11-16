#Custom CellsDataset class for dataloader to process images.
class CellsDataset(Dataset):
    def __init__(self, label_file, img_dir):
        #Initialise image directory
        self.img_dir = img_dir

        #Initialise labels
        try:
            self.img_labels = pd.read_csv(label_file) #Csv files.
        except:
            self.img_labels = label_file #Df files.
        
        # remove rows without an image path
        files = os.listdir(self.img_dir)
        self.img_labels = self.img_labels.loc[self.img_labels['Patient_ID'].isin(files)]

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        
        #Normalises images according to pretrained EfficientNet-B0 values
        transform_data = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        image = Image.open(img_path)
        image = transform_data(image)
        label = self.img_labels.iloc[idx, 2]

        return image, label
