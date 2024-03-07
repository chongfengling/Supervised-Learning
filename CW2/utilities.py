import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

def data_reader(file_path: str):
    """read and return data X, y

    Parameters
    ----------
    file_path : str
        file path

    Returns
    -------
    X: ndarray
        m*n data set
    y: ndarray
        m*1 label set
    """
    X, y = [], []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        row = np.asarray(line.rstrip().split())
        X.append(row[1:])
        y.append(int(float(row[0])))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)
    
def digit_visualization(grey_values: np.ndarray, label: np.ndarray, status: str='label'):
    """handwritten digit visualization for a single digit 

    Parameters
    ----------
    grey_values : np.ndarray
        1*256 grey values
    label : np.ndarray
        1*1 label of grey values
    status : str, optional
        (true) label or prediction, by default 'label'
    """
    image = grey_values.reshape(16,16)
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"{status}: {label}")
    plt.show()

def dataset_split(X: np.ndarray, y: np.ndarray, train_size: float=0.8):
    """split dataset X and y into train dataset and test dataset uniformly.

    Parameters
    ----------
    X : np.ndarray
        (m,n) dataset
    y : np.ndarray
        (m,1) label
    train_size : float, optional
        size of train dataset, by default 0.8

    Returns
    -------
    _type_
        _description_
    """
    X_train, X_test, y_train, y_test = [], [], [], []
    m = X.shape[0]
    no_of_train_set = int(m * train_size) # number of training data set
    index_train = np.asarray(random.sample(range(m), no_of_train_set)) # generate the index of training data
    for i in range(m):
        if i in index_train:
            X_train.append(X[i,:])
            y_train.append(y[i])
        else:
            X_test.append(X[i,:])
            y_test.append(y[i])
    return np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

def k_fold_cross_validation_datasets(k, training_dataset):
    """_summary_

    Args:
        k (int): number of folds for cross validation
        training_dataset (pd.DataFrame): (X,Y)

    Returns:
        list: a list contains k folds (pd.DataFrame)
    """
    m = training_dataset.shape[0] # number of examples in training set
    index = np.asarray(range(m))
    np.random.shuffle(index)
    folds = []

    for i in range(k):
        index_interval = int(m/k)
        start_index = i * index_interval
        end_index = (i+1) * index_interval -1
        if i == k-1:
            folds.append(training_dataset.loc[start_index:,:]) # collect the rest data
        else:
            folds.append(training_dataset.loc[start_index:end_index,:])
    return folds

def plot_bad_digits(sorted_Counter):
    i = 0
    while i < min(5, len(sorted_Counter)):
        string = sorted_Counter[i][0]
        y, X_string = string[0], string[2:-1].replace("'", '')
        X = np.asarray(X_string.split(','), dtype=float)
        digit_visualization(X,y)
        

        i += 1





if __name__ == '__main__':
    X, y = data_reader('data/dtrain123.dat')
    # print(X[0])
    # type(a)
    # index = 0
    # digit_visualization(X[index], y[index])

    print(X.shape)
    print(y.shape)

    # X_train,y_train,_,_ = dataset_split(X,y)
    # print(X_train.shape)

    # # digit_visualization(X[0], y[0])
    # digit_visualization(X_train[0], y_train[0])
    X_y = np.concatenate((X,y.reshape(y.shape[0],1)),axis=1)
    print(X_y.shape)
    X_y = pd.DataFrame(X_y)
    folds = k_fold_cross_validation_datasets(5, X_y)
    [print(x.shape) for x in folds]
