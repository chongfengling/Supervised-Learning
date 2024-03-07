from utilities import *
from tqdm import tqdm
from scipy.spatial.distance import cdist
import pandas as pd
from collections import Counter

class Perceptron:
    
    def __init__(self, model: str="Binary", kernel: str="Polynomial", d: int=0, c: int=0, epoch_num: int=2) -> None:
        # self.weight = None
        self.alpha = None
        self.best_alpha = None
        self.best_train_error = None
        self.model = model
        self.kernel = kernel
        self.epoch_no_improvement = epoch_num # 

        if self.kernel == "Polynomial":
            self.d = d
            self.c = None
        elif self.kernel == "Gaussian":
            self.d = None
            self.c = c

    def update(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray=None, y_val: np.ndarray=None):
        epoch = 0
        epoch_no_improvement = 0

        if self.model == 'Binary':
            best_val_error = 1

            self.alpha = np.zeros(X_train.shape[0])
            train_error = self.__binary_classification(X_train, y_train)
            self.best_alpha = self.alpha

            # while epoch_no_improvement < self.epoch_no_improvement:

            #     # update alpha and train_error
            #     train_error = self.__binary_classification(X_train, y_train)
            #     # update val_error
            #     val_error = self.__binary_prediction(X_train, X_val, y_val, val=True)

            #     if val_error < best_val_error:
            #         self.best_alpha = self.alpha
            #         best_val_error = val_error
            #         self.best_train_error = train_error
            #     else:
            #         epoch_no_improvement += 1
            #     epoch += 1

            # self.epoch = epoch
                # print(f"epoch {epoch}, train error: {train_error:.5f}, val error: {val_error:.5f}, best val error: {best_val_error:.5f}")

        if self.model == "Multiple":
            best_val_error = 1
            self.classes = np.unique(y_train) # sort or not?

            self.alpha = np.zeros((len(self.classes), X_train.shape[0]))

            if (X_val is not None) and (y_val is not None):

                while epoch_no_improvement < self.epoch_no_improvement:
                    train_error = self.__multiple_classification(X_train, y_train)
                    val_error = self.__multiple_prediction(X_train, X_val, y_val, val=True)

                    if val_error < best_val_error:
                        self.best_alpha = self.alpha
                        best_val_error = val_error
                        self.best_train_error = train_error
                    else:
                        epoch_no_improvement += 1
                    epoch += 1

                self.epoch = epoch

            else:

                train_error = self.__multiple_classification(X_train, y_train)
                self.best_alpha = self.alpha


    def predict(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        if self.model == 'Binary':
            self.pred_error = self.__binary_prediction(X_train, X_test, y_test)
        elif self.model == 'Multiple':
            self.pred_error = self.__multiple_prediction(X_train, X_test, y_test)

    def __polynomial_kernel(self, X: np.ndarray, X_primed: np.ndarray=None) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        X : np.ndarray
            (m,n) array
        X_primed : np.ndarray
            X or (m_primed,n) array

        Returns
        -------
        np.ndarray
            X @ X.T kernel matrix
        """
        if X_primed is None:
            return np.power(np.dot(X, X.T), self.d)
        else:
            return np.power(np.dot(X, X_primed.T), self.d)

    def __gaussian_kernel(self, X: np.ndarray, X_primed: np.ndarray=None) -> np.ndarray:
        """_summary_

        Parameters
        ----------
        X : np.ndarray (m_1, n)
            _description_
        X_primed : np.ndarray, optional (m_2, n)
            _description_, by default None

        Returns
        -------
        np.ndarray
            _description_
        """
        if X_primed is None:
            K = cdist(X, X, 'euclidean')
            K = np.exp(-self.c * (K ** 2))
        else:
            assert(X.shape[1] == X_primed.shape[1])
            K = cdist(X, X_primed, 'euclidean')
            K = np.exp(-self.c * (K ** 2))
        return K

    def __sign_function(self, constant: float) -> int:
        if constant > 0:
            return 1
        elif constant <=0:
            return -1

    def __binary_classification(self, X: np.ndarray, y: np.ndarray):
        """_summary_

        Parameters
        ----------
        X : np.ndarray
            _description_
        y : np.ndarray
            -1 or 1
        """
        # self.alpha = np.zeros(y.shape[0])
        train_error = 0

        if self.kernel == "Polynomial":
            kernel_matrix = self.__polynomial_kernel(X)
        elif self.kernel == "Gaussian":
            kernel_matrix = self.__gaussian_kernel(X)

        # y = np.asarray([-1 if i == y[0] else 1 for i in y])

        for m in range(X.shape[0]):
            y_m_pred = self.__sign_function(np.dot(self.alpha, kernel_matrix[:,m]))
            if y_m_pred != y[m]:
                self.alpha[m] = y[m]
                train_error += 1
        return train_error / X.shape[0]

    def __multiple_classification(self, X: np.ndarray, y: np.ndarray):
        """_summary_

        Parameters
        ----------
        X : np.ndarray
            _description_
        y : np.ndarray
            _description_

        Returns
        -------
        _type_
            _description_
        """
        train_error = 0

        if self.kernel == "Polynomial":
            kernel_matrix = self.__polynomial_kernel(X)
        elif self.kernel == "Gaussian":
            kernel_matrix = self.__gaussian_kernel(X)

        for m in range(X.shape[0]):
            max_confidence = -1e20
            y_m_confidence = np.dot(self.alpha, kernel_matrix[:,m]) # (k,) array
            y_true = y[m]
            for i in range(len(self.classes)):
                y_tmp = [1 if y_true == self.classes[i] else -1]

                # update alpha
                if y_tmp[0] * y_m_confidence[i] <= 0:
                    self.alpha[i, m] += -1 * self.__sign_function(y_m_confidence[i])
                # update predicted y
                if y_m_confidence[i] > max_confidence:
                    max_confidence = y_m_confidence[i]
                    y_pred = self.classes[i]
            # compare the final predicted y and true y
            if y_pred != y_true:
                train_error += 1
        
        return train_error / X.shape[0]

    def __binary_prediction(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, val=False):
        # y_pred = []
        test_error = 0

        if self.kernel == 'Polynomial':
            kernel_matrix = self.__polynomial_kernel(X_train, X_test)
        elif self.kernel == "Gaussian":
            pass

        if val:
            alpha = self.alpha
        else:
            alpha = self.best_alpha

        for m in range(X_test.shape[0]):
            # using corresponding alpha
            y_m_pred = self.__sign_function(np.dot(alpha, kernel_matrix[:,m]))

            if y_m_pred != y_test[m]:
                test_error += 1
            # y_pred.append(y_m_pred)

        # print(test_error / X_test.shape[0])
        return test_error / X_test.shape[0]

    def __multiple_prediction(self, X_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, val=False):

        test_error = 0
        y_pred_list = []

        if self.kernel == 'Polynomial':
            kernel_matrix = self.__polynomial_kernel(X_train, X_test)
        elif self.kernel == "Gaussian":
            kernel_matrix = self.__gaussian_kernel(X_train, X_test)

        if val:
            alpha = self.alpha
        else:
            alpha = self.best_alpha

        for m in range(X_test.shape[0]):
            max_confidence = -1e20
            y_m_confidence = np.dot(alpha, kernel_matrix[:,m]) # (k,) array
            y_true = y_test[m]

            for i in range(len(self.classes)):

                # update predicted y
                if y_m_confidence[i] > max_confidence:
                    max_confidence = y_m_confidence[i]
                    y_pred = self.classes[i]

            y_pred_list.append(y_pred)
            if y_pred != y_true:
                test_error += 1

        # for confusion matrix analysis
        self.y_pred = np.asarray(y_pred_list)
        self.y_true = y_test
        self.X_test = X_test

        return test_error / X_test.shape[0]

    def analysis(self):
        return self.best_train_error, self.pred_error

    def confusion_matrix_and_wrong_pred_index(self):

        true_pred_list, X_error, X_error_label = [], [], []
        # [true_pred_list.append((true, pred)) for true, pred in zip(self.y_true, self.y_pred) if true != pred]
        for  index, (true, pred) in enumerate(zip(self.y_true, self.y_pred)):
            if true != pred:
                true_pred_list.append((true, pred))
                # print(true_pred_list)
                X_error.append(str(true) + [str(["".join(item) for item in self.X_test[index,:].astype(str)])][0]) # "2['-1.0',...,'1.0']"
                # X_error_label.append(true)
        return true_pred_list, X_error

# tested   
def experiment_1():
    # load data (X,y)
    X, y = data_reader('data/zipcombo.dat')

    summary_train, summary_test = [], []
    df = pd.DataFrame(columns=range(8))
    # parameter of the polynomial kernel
    for d in tqdm(range(1,8)):
        train_error_d, test_error_d = [], []
        for _ in range(20):
            X_train, y_train, X_test, y_test = dataset_split(X, y, train_size=0.8)
            # separate validation data from training data to determine the hyperparameter: number of epoch
            X_train, y_train, X_val, y_val = dataset_split(X_train, y_train, train_size=0.9)

            model = Perceptron(d=d, model="Multiple")
            model.update(X_train, y_train, X_val, y_val)
            model.predict(X_train, X_test, y_test)

            train_error, test_error = model.analysis()
            train_error_d.append(train_error)
            test_error_d.append(test_error)
            # print(f'{train_error:.5f}, {test_error:.5f}')
        summary_train.append(train_error_d)
        summary_test.append(test_error_d)
    summary_train, summary_test = np.asarray(summary_train), np.asarray(summary_test)

    tr_m, tr_s = summary_train.mean(axis=1), summary_train.std(axis=1)
    te_m, te_s = summary_test.mean(axis=1), summary_test.std(axis=1)

    row_1, row_2 = [], []
    for d, (i,j) in enumerate(zip(tr_m, tr_s)):
        row_1.append(str(round(i, 4)) + ' +- ' + str(round(j,4)))

    for d, (i,j) in enumerate(zip(te_m, te_s)):
        row_2.append(str(round(i, 4)) + ' +- ' + str(round(j,4)))
    df.loc['train error'] = row_1
    df.loc['test error'] = row_2
    print(df)
# tested
def experiment_2():
    X, y = data_reader('data/zipcombo.dat')
    best_d_list, test_error_best_d_list = [], []
    for _ in tqdm(range(20)):
        X_train, y_train, X_test, y_test = dataset_split(X, y, train_size=0.8)
        test_error_summary = {}
        X_y = pd.DataFrame(np.concatenate((X_train,y_train.reshape(y_train.shape[0],1)),axis=1))
        for d in range(1,8):
            folds = k_fold_cross_validation_datasets(k=5, training_dataset=X_y)
            test_error_d = []
            for i in range(5):
                X_y_test_fold = folds[i]
                X_y_train_fold = pd.concat([folds[j] for j in range(len(folds)) if j != i])

                X_train_fold = np.asarray(X_y_train_fold.iloc[:,:-1])
                y_train_fold = np.asarray(X_y_train_fold.iloc[:,-1])
                X_test_fold = np.asarray(X_y_test_fold.iloc[:,:-1])
                y_test_fold = np.asarray(X_y_test_fold.iloc[:,-1])

                model = Perceptron(model="Multiple", kernel="Polynomial", d=d)
                model.update(X_train_fold, y_train_fold)
                model.predict(X_train_fold, X_test_fold, y_test_fold)

                _, test_error = model.analysis()
                test_error_d.append(test_error)

            test_error_summary[d] = np.mean(test_error_d)
        
        best_d, _ = min(test_error_summary.items(),key=lambda x:x[1])

        model = Perceptron(model="Multiple", kernel="Polynomial", d=best_d)
        model.update(X_train, y_train)
        model.predict(X_train, X_test, y_test)
        _, best_d_test_error = model.analysis()
        best_d_list.append(best_d)
        test_error_best_d_list.append(best_d_test_error)

    print(f"mean +_ std of test error is {np.mean(test_error_best_d_list):.4f} +_ {np.std(test_error_best_d_list):.4f}")
    print(f"mean +_ std of d_star is {np.mean(best_d_list):.4f} +_ {np.std(best_d_list):.4f}")

def experiment_3_and_4():
    X, y = data_reader('data/zipcombo.dat')
    summary, X_error = [], []
    confusion_matrix = np.zeros((20,10,10))
    for run in tqdm(range(20)):
        X_train, y_train, X_test, y_test = dataset_split(X, y, train_size=0.8)
        test_error_summary = {}
        X_y = pd.DataFrame(np.concatenate((X_train,y_train.reshape(y_train.shape[0],1)),axis=1))
        for d in range(1,8):
            folds = k_fold_cross_validation_datasets(k=5, training_dataset=X_y)
            test_error_d = []
            for i in range(5):
                X_y_test_fold = folds[i]
                X_y_train_fold = pd.concat([folds[j] for j in range(len(folds)) if j != i])

                X_train_fold = np.asarray(X_y_train_fold.iloc[:,:-1])
                y_train_fold = np.asarray(X_y_train_fold.iloc[:,-1])
                X_test_fold = np.asarray(X_y_test_fold.iloc[:,:-1])
                y_test_fold = np.asarray(X_y_test_fold.iloc[:,-1])

                model = Perceptron(model="Multiple", kernel="Polynomial", d=d)
                model.update(X_train_fold, y_train_fold)
                model.predict(X_train_fold, X_test_fold, y_test_fold)

                _, test_error = model.analysis()
                test_error_d.append(test_error)

            test_error_summary[d] = np.mean(test_error_d)
        
        best_d, _ = min(test_error_summary.items(),key=lambda x:x[1])

        model = Perceptron(model="Multiple", kernel="Polynomial", d=best_d)
        model.update(X_train, y_train)
        model.predict(X_train, X_test, y_test)
        _, best_d_test_error = model.analysis()
        summary.append((best_d, best_d_test_error))

        counter = Counter(y_test) # number of digit 

        confusion_record, X_error_of_d_star = model.confusion_matrix_and_wrong_pred_index()

        for i,j in confusion_record:
            # print((i,j))
            confusion_matrix[run,i,j] += (1/ counter[i])
        
        X_error += X_error_of_d_star
        # X_error_label += error_label

    # experiment 3
    confusion_matrix_mean = np.mean(confusion_matrix, axis=0)
    confusion_matrix_std = np.std(confusion_matrix, axis=0)
    # matrix[i,j], i is the true label and j is the wrong pred label
    # print("confusion error matrix (mean)")
    df1 = pd.DataFrame(data = confusion_matrix_mean, columns=range(10))
    # print(df1)
    # print(confusion_matrix_mean)
    # print("-" * 15)
    # print("confusion error matrix (std)")
    df2 = pd.DataFrame(data = confusion_matrix_std, columns=range(10))
    # print(df2)
    # print(confusion_matrix_std)

    # experiment 4
    sorted_counter_X_error = sorted(Counter(X_error).items(), key = lambda x:x[1], reverse=True)

    return df1, df2, sorted_counter_X_error
    # plot_bad_digits(sorted_counter_X_error)

    # return confusion_matrix_mean, confusion_matrix_std, X_error



def experiment_5_1():
    # load data (X,y)
    X, y = data_reader('data/zipcombo.dat')

    summary_train, summary_test = [], []

    s_val = [1e-2, 1e-1, 1e0, 1e1, 1e2]

    # parameter of the polynomial kernel
    for c in tqdm(s_val):
        train_error_d, test_error_d = [], []
        for _ in range(20):
            X_train, y_train, X_test, y_test = dataset_split(X, y, train_size=0.8)
            # separate validation data from training data to determine the hyperparameter: number of epoch
            X_train, y_train, X_val, y_val = dataset_split(X_train, y_train, train_size=0.9)

            model = Perceptron(c=c, model="Multiple", kernel="Gaussian")
            model.update(X_train, y_train, X_val, y_val)
            model.predict(X_train, X_test, y_test)

            train_error, test_error = model.analysis()
            train_error_d.append(train_error)
            test_error_d.append(test_error)
            # print(f'{train_error:.5f}, {test_error:.5f}')
        summary_train.append(train_error_d)
        summary_test.append(test_error_d)
    summary_train, summary_test = np.asarray(summary_train), np.asarray(summary_test)

    tr_m, tr_s = summary_train.mean(axis=1), summary_train.std(axis=1)
    te_m, te_s = summary_test.mean(axis=1), summary_test.std(axis=1)

    for d, (i,j) in enumerate(zip(tr_m, tr_s)):
        print(f"mean +_ std of train error with  c={d+1} is {i:.4f} +_ {j:.4f}")

    for d, (i,j) in enumerate(zip(te_m, te_s)):
        print(f"mean +_ std of test error with  c={d+1} is {i:.4f} +_ {j:.4f}")

def experiment_5_2():
    S_val = np.linspace(0.001,0.1,5)
    X, y = data_reader('data/zipcombo.dat')
    best_d_list, test_error_best_d_list = [], []
    for _ in tqdm(range(20)):
        X_train, y_train, X_test, y_test = dataset_split(X, y, train_size=0.8)
        test_error_summary = {}
        X_y = pd.DataFrame(np.concatenate((X_train,y_train.reshape(y_train.shape[0],1)),axis=1))
        for c in range(S_val):
            folds = k_fold_cross_validation_datasets(k=5, training_dataset=X_y)
            test_error_d = []
            for i in range(5):
                X_y_test_fold = folds[i]
                X_y_train_fold = pd.concat([folds[j] for j in range(len(folds)) if j != i])

                X_train_fold = np.asarray(X_y_train_fold.iloc[:,:-1])
                y_train_fold = np.asarray(X_y_train_fold.iloc[:,-1])
                X_test_fold = np.asarray(X_y_test_fold.iloc[:,:-1])
                y_test_fold = np.asarray(X_y_test_fold.iloc[:,-1])

                model = Perceptron(model="Multiple", kernel="Gaussian", c=c)
                model.update(X_train_fold, y_train_fold)
                model.predict(X_train_fold, X_test_fold, y_test_fold)

                _, test_error = model.analysis()
                test_error_d.append(test_error)

            test_error_summary[c] = np.mean(test_error_d)
        
        best_d, _ = min(test_error_summary.items(),key=lambda x:x[1])

        model = Perceptron(model="Multiple", kernel="Gaussian", d=best_d)
        model.update(X_train, y_train)
        model.predict(X_train, X_test, y_test)
        _, best_d_test_error = model.analysis()
        best_d_list.append(best_d)
        test_error_best_d_list.append(best_d_test_error)

    print(f"mean +_ std of test error is {np.mean(test_error_best_d_list):.4f} +_ {np.std(test_error_best_d_list):.4f}")
    print(f"mean +_ std of c_star is {np.mean(best_d_list):.4f} +_ {np.std(best_d_list):.4f}")

def experiment_6_1():
    return 0

def experiment_6_2():
    return 0



if __name__ == '__main__':
    # experiment_5_1()
    experiment_3_and_4()