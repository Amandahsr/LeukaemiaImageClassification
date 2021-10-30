#Initialise testing set files
test_path = "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/held_out_test/test_labels.csv"
test_df = pd.read_csv(test_path)[["Patient_no", "Patient_ID", "labels"]]
test_set = CellsDataset(test_df, "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/held_out_test/test_images")

#Dataloader for testing input.
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#Test models.
base_testStats = test_model(test_loader, base_ensembl_model)
mpl_testStats = test_model(test_loader, mpl_ensembl_model)

