# Split the final pandas DataFrame into training and evaluation sets
def split_train_eval(final_dataset):
    # CODE HERE
    #from sklearn.model_selection import train_test_split
    #return train_test_split(final_dataset, test_size=0.1)
    final_dataset = final_dataset.sample(frac=1)
    eval_size = len(final_dataset) // 10
    eval_set = final_dataset.iloc[:eval_size]
    train_set = final_dataset.iloc[eval_size:]
    return train_set, eval_set

