import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sk
from sklearn import svm
from copy import copy, deepcopy
from sklearn.model_selection import StratifiedKFold
from mpl_toolkits.mplot3d import Axes3D



def gen_sample(mus, sigmas):
    sample =np.random.multivariate_normal(mus, sigmas)
    return sample[0], sample[1]


def generate_samples(N, priors, mus, sigmas):
    # Assign samples and generate values
    samples = []
    index = 0

    for i in range(N):
        sample = [0, 0, 0]
        rand = np.random.uniform()
        #class 0
        if rand > priors[0]:
            sample[2] = 1
            sample[0], sample[1] = gen_sample(mus[1], sigmas[1])

        #class 1
        else:

            sample[0], sample[1] = gen_sample(mus[0], sigmas[0])
        samples.append(sample)


    return samples



def eval_classifications(assignments, classifications):
    """Compare the assignments and classifications to find errors"""
    correct = 0
    for i in range(len(assignments)):
        #Real val is pos
        if assignments[i] == classifications[i]:
            correct += 1

    return correct/len(assignments)


class Linear_SVM_Model():
    def __init__(self, Dtrain, penalty):
        self.classifier = svm.LinearSVC(C=penalty)
        self.labels = []
        self.dtrain = []
        for i in range(len(Dtrain)):

            self.dtrain.append(Dtrain[i][0:2])
            self.labels.append(Dtrain[i][2])

    def fit_model(self):
        self.classifier.fit(self.dtrain, np.array(self.labels).reshape(-1,1))

    def validate(self, Dval):
        self.dval = Dval
        self.predictions = []
        for i in range(len(Dval)):
            self.predictions.append(self.classifier.predict(np.array(Dval[i][0:2]).reshape(1,-1))[0])

    def eval_performance(self):
        actual = [sample[2] for sample in self.dval]
        perf = eval_classifications(actual, self.predictions)
    #    print(perf)
        return perf

    def get_predictions(self):
        return self.predictions


class Gaussian_SVM_Model():
    def __init__(self, Dtrain, penalty, kernel_scale):
        self.classifier = svm.SVC(kernel='rbf', gamma=kernel_scale, C=penalty)
        self.labels = []
        self.dtrain = []
        for i in range(len(Dtrain)):
            self.dtrain.append(Dtrain[i][0:2])
            self.labels.append(Dtrain[i][2])

    def fit_model(self):
        self.classifier.fit(self.dtrain, np.array(self.labels).reshape(-1, 1))

    def validate(self, Dval):
        self.dval = Dval
        self.predictions = []
        for i in range(len(Dval)):
            self.predictions.append(self.classifier.predict(np.array(Dval[i][0:2]).reshape(1, -1))[0])

    def eval_performance(self):
        actual = [sample[2] for sample in self.dval]
        perf = eval_classifications(actual, self.predictions)
        #    print(perf)
        return perf

    def get_predictions(self):
        return self.predictions

def generate_subsets(data, k):
    validation_sets = []
    training_sets = []
    kf = StratifiedKFold(k)
    X = [[entry[0], entry[1]] for entry in data]
    y = [entry[2] for entry in data]
    for train_index, test_index in kf.split(X, y):
        validation_sets.append([X[index]+[y[index]] for index in test_index])
        training_sets.append([X[index]+[y[index]] for index in train_index])
    return validation_sets, training_sets


def select_linear_params(data, k, penalty_candidates):
    #k is the cross validation parameter
    validation_sets, training_sets = generate_subsets(data, k)
    penalty_performance_measures = []
    for j in range(len(penalty_candidates)):
        accuracies = 0
        for i in range(k):
            Dval = validation_sets[i]
            Dtrain = training_sets[i]
            model = Linear_SVM_Model(Dtrain, penalty_candidates[j])
            model.fit_model()
            model.validate(Dval)
            performance = model.eval_performance()

            accuracies += performance
            del model
        penalty_performance_measures.append(float(accuracies/k))

    return penalty_performance_measures


def select_gaussian_params(data, k, penalty_candidates, scale_candidates):
    #Will return a matrix with performance[j][k] where j is penalty, k is scale
    #k is the cross validation parameter
    validation_sets, training_sets = generate_subsets(data, k)
    performance_measures = []
    for j in range(len(penalty_candidates)):
        penalty_performance_varying_scales = []
        for x in range(len(scale_candidates)):
            accuracies = 0
            for i in range(k):
                Dval = validation_sets[i]
                Dtrain = training_sets[i]
                model = Gaussian_SVM_Model(Dtrain, penalty_candidates[j], scale_candidates[x])
                model.fit_model()
                model.validate(Dval)
                accuracies += model.eval_performance()
          #  penalty_performance_varying_scales.append(accuracies/k)
            performance_measures.append(accuracies/k)

    return performance_measures

    
def linear_performance_plot(performance_measures, penalty_candidates):
    max_idx = performance_measures.index(max(performance_measures))
    plt.scatter(penalty_candidates, performance_measures)
    plt.scatter([penalty_candidates[max_idx]], [performance_measures[max_idx]], c="red", s=20)
    plt.xlabel('Penalty Value')
    plt.ylabel('Accuracy')
    print('Best linear')
    print('penalty', penalty_candidates[max_idx])
    print('accuracy', performance_measures[max_idx])
    plt.title('Penalty vs. Accuracy')
    plt.show()



def gaussian_performance_plot(performance_measures, penalty_candidates, scale_candidates):
    x = []
    y = []
    z = []
    index = 0
    print(len(performance_measures))
    maxidx = 0
    for i in penalty_candidates:
        for j in scale_candidates:
            x.append(i)
            y.append(j)
            z.append(performance_measures[index])
            if performance_measures[index] == max(performance_measures):
                maxidx = index
            index += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, zs = z )
    ax.scatter3D([x[maxidx]], [y[maxidx]], zs=[performance_measures[maxidx]], c="red", s=20)
    print('Best performance')
    print('penalty', x[maxidx])
    print('scale', y[maxidx])
    print('performance', performance_measures[maxidx])
    ax.set_xlabel('Penalty Candidates')
    ax.set_ylabel('Scale Candidates')
    ax.set_zlabel('Performance')
    plt.title('Penalty vs. Accuracy')
    plt.show()


def plot_linear(data, penalty):
    model = Linear_SVM_Model(data, penalty)
    model.fit_model()
    model.validate(data)
    predictions = model.get_predictions()
    list0x = []
    list0y = []
    list1x = []
    list1y = []
    for i in range(len(data)):
        if predictions[i] == 0:
            list0x.append(data[i][0])
            list0y.append(data[i][1])
        else:
            list1x.append(data[i][0])
            list1y.append(data[i][1])
    plt.scatter(list0x, list0y, c="red", marker ="x")
    plt.scatter(list1x, list1y, c="blue", marker="o")
    plt.show()


def plot_gaussian(data, penalty, scale):
    model = Gaussian_SVM_Model(data, penalty, scale)
    model.fit_model()
    model.validate(data)
    predictions = model.get_predictions()
    list0x = []
    list0y = []
    list1x = []
    list1y = []
    for i in range(len(data)):
        if predictions[i] == 0:
            list0x.append(data[i][0])
            list0y.append(data[i][1])
        else:
            list1x.append(data[i][0])
            list1y.append(data[i][1])
    plt.scatter(list0x, list0y, c="red", marker ="x")
    plt.scatter(list1x, list1y, c="blue", marker="o")
    plt.show()






if __name__ == "__main__":

    n = 2  #Dimensions per feature
    N = 1000 #Number of samples
    mu0 = np.array([-1, 0])
    mu1 = np.array([1, 0])
    sigma0 = np.array([[2, 0], [0, 1]])
    sigma1 = np.array([[1, 0], [0, 4]])

    mus = [mu0, mu1]
    sigmas = [sigma0, sigma1]
    priors = [0.35, 0.65]
    k = 10
    penalty_candidates = np.linspace(0.01, 1000, 20)
    scale_candidates = np.linspace(0.01, 1000, 20)
    data = generate_samples(N, priors, mus, sigmas)
 #   penalty_performance_measures = select_linear_params(data, k, penalty_candidates)
    #print(penalty_performance_measures)
    #linear_performance_plot(penalty_performance_measures, penalty_candidates)
    plot_linear(data, 42.111)
    plot_gaussian(data, 31.586, 0.01)


    performance_measures = select_gaussian_params(data,k, penalty_candidates, scale_candidates)
    gaussian_performance_plot(performance_measures, penalty_candidates, scale_candidates)