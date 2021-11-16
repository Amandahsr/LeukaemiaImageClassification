# Define stratified K-fold cross-validation function.
strat_kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=CV_states)

#Obtain indexes for stratified k fold.
trainLabel_df = pd.read_csv(train_labels)
X = np.zeros(len(trainLabel_df))
y = trainLabel_df['labels']
