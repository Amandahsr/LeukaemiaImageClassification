#Initialise labels file.
label_csv = "/content/ZB4171_LeukemiaImageClassification-Ongoing-/Data_main/train_labels.csv"

# Define stratified K-fold cross-validation function.
strat_kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=CV_states)

#Obtain indexes for stratified k fold.
label_df = pd.read_csv(label_csv)
X = np.zeros(len(label_df))
y = label_df['labels']
