import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from pandas import read_csv
import seaborn as sns


class LinearRegressor(object):
    def __init__(self) -> None:
        # variables to store mean and standard deviation for each feature
        self.mu = []
        self.std = []
        self.theta = []
        self.costs = []

    def load_data(self, data):
        data = np.array(data, dtype=float)
        self.normalize(data)
        return data[:, 1:], data[:, 0]

    def normalize(self, data):
        for i in range(1, data.shape[1]):
            mean = np.mean(data[:, i])
            std = np.std(data[:, i])
            data[:, i] = ((data[:, i] - mean) / std)
            self.mu.append(mean)
            self.std.append(std)

    def h(self, x, theta):
        return np.matmul(x, theta)

    def cost_function(self, x, y, theta):
        return ((self.h(x, theta) - y).T @ (self.h(x, theta) - y)) / (y.shape[0])

    def gradient_descent(self, x, y, theta, learning_rate=0.1, num_epochs=10):
        # number of samples
        m = x.shape[0]
        J_all = []
        for _ in range(num_epochs):
            h_x = self.h(x, theta)
            gradient = (1/m)*(x.T@(h_x - y))
            theta = theta - learning_rate*gradient
            cost = self.cost_function(x, y, theta)
            J_all.append(cost)

        return theta, J_all

    def plot_cost(self):
        # for testing and plotting cost
        n_epochs = []
        jplot = []
        count = 0
        for i in self.costs:
            jplot.append(i[0][0])
            n_epochs.append(count)
            count += 1
        jplot = np.array(jplot)
        n_epochs = np.array(n_epochs)

        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.plot(n_epochs, jplot, 'm', linewidth="2")
        plt.title('Training Loss for linear regression')
        plt.show()

    def train(self, x, y, lr, num_epochs):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        y = np.reshape(y, (x.shape[0], 1))
        theta = np.zeros((x.shape[1], 1))
        self.theta, J_all = self.gradient_descent(x, y, theta, lr, num_epochs)
        self.costs = J_all
        # returns final cost
        return J_all

    def predict(self, x):
        x = np.hstack((np.ones((x.shape[0], 1)), x))
        for i in range(1, x.shape[1]):
            x[:, i] = (x[:, i] - self.mu[i-1]) / self.std[i-1]
        y = np.matmul(x, self.theta)
        return y

    # Save the model as a pickle file
    def save(self, name):
        with open('models/' + name + '.pkl', 'wb') as f:
            pickle.dump([self.mu, self.std, self.theta], f)

    # Load the model from a trained pickle file
    def load(self, name):
        with open('models/' + name + '.pkl', 'rb') as f:
            model = pickle.load(f)
            self.mu = list(model[0])
            self.std = list(model[1])
            self.theta = list(model[2])

# gradient boost tree regressor #


class GradientBoostTreeRegressor(object):
    def __init__(self, n_elements=100, max_depth=1):
        self.max_depth = max_depth
        self.n_elements = n_elements
        self.f = []
        self.regions = []
        self.gammas = []
        self.model_weights = []
        self.mean_loss = []
        self.e0 = 0

    # destructor
    def __del__(self):
        del self.max_depth
        del self.n_elements
        del self.f
        del self.regions
        del self.gammas
        del self.mean_loss
        del self.e0

    # private function to group data points & compute gamma parameters
    def __compute_gammas(self, yp, y_train, e, split, index):
        # initialise global gamma array
        gamma_jm = np.zeros((y_train.shape[0]))
        # iterate through each unique predicted value/region
        regions = np.unique(yp)
        gamma = {}
        for r in regions:
            # compute index for r
            idx = yp == r
            # isolate relevant data points
            e_r = e[idx]
            y_r = y_train[idx]
            # compute the optimal gamma parameters for region r
            gamma_r = np.median(y_r - e_r)
            # populate the global gamma array
            gamma_jm[idx] = gamma_r
            # set the unique region <-> gamma pairs
            gamma[r] = gamma_r
        # append the regions to internal storage
        if split == 0:
            self.regions.append(regions)
        else:
            self.regions[index] = regions
        return (gamma_jm, gamma)

    # public function to train the ensemble
    def fit(self, x_train, y_train, split):
        # reset the internal class members
        # self.f = []
        # self.regions = []
        # self.model_weights = []
        self.mean_loss = []
        # initialise the ensemble & store initialisation
        e0 = np.median(y_train)
        self.e0 = np.copy(e0)
        e = np.ones(y_train.shape[0]) * e0
        # loop through the specified number of iterations in the ensemble
        for i in range(self.n_elements):
            # store mae loss
            self.mean_loss.append(np.mean(np.abs(y_train - e)))
            # compute the gradients of our loss function
            g = np.sign(y_train - e)
            # initialise a weak learner & train
            if split == 0:
                model = DecisionTreeRegressor(max_depth=self.max_depth)
            else:
                # print("index ", i, len(self.f))
                model = self.f[i]
            model.fit(x_train, g)
            # compute optimal gamma coefficients
            yp = model.predict(x_train)
            gamma_jm, gamma = self.__compute_gammas(yp, y_train, e, split, i)
            # update the ensemble
            e += gamma_jm
            # store trained ensemble elements
            if split == 0:
                self.f.append(model)
                self.gammas.append(gamma)
            else:
                self.f[i] = model
                self.gammas[i] = gamma

    # public function to generate predictions
    def predict(self, x_test):
        # initialise predictions
        y_pred = np.ones(x_test.shape[0]) * np.copy(self.e0)
        # cycle through each element in the ensemble
        for model, gamma, regions in zip(self.f, self.gammas, self.regions):
            # produce predictions using model
            y = model.predict(x_test)
            # cycle through each unique leaf node for model m
            for r in regions:
                # updates for region r
                idx = y == r
                y_pred[idx] += gamma[r]
                # return predictions
        return y_pred

    # public function to return mean training loss
    def get_loss(self):
        return self.mean_loss

    # public function to return model parameters
    def get_params(self):
        return {'n_elements': self.n_elements,
                'max_depth': self.max_depth}

    # Save the model as a pickle file
    def save(self, name):
        with open('models/' + name + '.pkl', 'wb') as f:
            pickle.dump([self.f, self.gammas, self.regions], f)

    # Load the model from a trained pickle file
    def load(self, name):
        with open('models/' + name + '.pkl', 'rb') as f:
            model = pickle.load(f)
            self.f = list(model[0])
            self.gammas = list(model[1])
            self.regions = list(model[2])


def create_features(data, starting_index):
    """
    Create time series features
    :param data: pandas dataframe to be updated
    :param starting_index: the index of first numeric day for the dataset
    :return: data
    """
    data = data.copy()
    data['index'] = [x for x in range(starting_index, starting_index + len(data.index))]
    data['dayofweek'] = data.index.dayofweek
    data['month'] = data.index.month
    data['quarter'] = data.index.quarter
    return data


def plot_feature(data, feature):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.boxplot(data=data, x=feature, y='Receipt_Count')
    ax.set_title('Receipt Count by ' + feature)
    plt.show()


def gbtr_fit(n_elements, max_depth, data, index, tscv=True):
    rgr = GradientBoostTreeRegressor(n_elements=n_elements, max_depth=max_depth)
    loss = 0
    if tscv:
        n_samples = data.values.shape[0]
        train_samples = math.floor(n_samples*0.2)
        test_samples = math.floor(n_samples*0.05)
        for i in range(0, n_samples-train_samples-test_samples):
            rgr.fit(data.values[0:i+train_samples, index:], data.values[0:i+train_samples, 0], i)
            loss = rgr.get_loss()
    return rgr, loss


def train():

    data = read_csv('./data/data_daily.csv', header=0, parse_dates=[0], index_col=0)
    # The pandas dataframe contains the receipt counts as the first column
    # and 4 new features are added in the subsequent columns
    data = create_features(data, 1)

    # Get a basic idea of how each feature is varying
    # plot_feature(data, 'dayofweek')
    # plot_feature(data, 'month')
    # plot_feature(data, 'quarter')

    # Generate a linear regressor object
    # This regressor is responsible for calculating the general trend of linear
    # increase in receipt count per day and uses only index as the input feature
    l_rgr = LinearRegressor()
    x, y = l_rgr.load_data(data.values[:, 0:2])
    learning_rate = 0.1
    num_epochs = 50
    l_rgr.train(x, y, learning_rate, num_epochs)
    l_rgr.save('lr')

    # Prediction using the trained linear regression model
    y_pred_lr = l_rgr.predict(np.array(data.values[:, 1:2], dtype=float))

    # Update the dataset by subtracting predictions from linear regression
    # This leaves us with dataset where we can find seasonal patterns if any
    data['Receipt_Count'] = data['Receipt_Count'] - y_pred_lr.reshape(365, )
    gb_feature_start_index = 1

    rgr, loss = gbtr_fit(100, 4, data, gb_feature_start_index)
    rgr.save('gbt')


def run_inference():
    l_rgr = LinearRegressor()
    l_rgr.load('lr')
    gb_feature_start_index = 1
    rgr = GradientBoostTreeRegressor(n_elements=100, max_depth=4)
    rgr.load('gbt')

    data = read_csv('./data/data_daily.csv', header=0, parse_dates=[0], index_col=0)
    index = pd.date_range('2022-01-01', '2022-12-31')
    data_new = read_csv('./data/data_daily.csv', header=0, parse_dates=[0], index_col=0)
    data_new = data_new.drop(columns= 'Receipt_Count')
    data_new.index = index
    data_new = create_features(data_new, 366)

    pred_lr = l_rgr.predict(data_new.values[:, 0:1])
    data_new['index'] = data_new['index'] - 365
    pred_gb = rgr.predict(np.array(data_new.values[:, gb_feature_start_index-1:]))
    data_new['Receipt_Count_Predicted'] = pred_lr.reshape(pred_lr.shape[0], 1) + pred_gb.reshape(pred_lr.shape[0], 1)
    data_new['Receipt_Count_Predicted'] = data_new['Receipt_Count_Predicted'].astype(int)
    fig, ax = plt.subplots(figsize=(10,5))
    data.plot(ax=ax, y='Receipt_Count', label='Original Data')
    data_new.plot(ax=ax, y='Receipt_Count_Predicted', label='Predicted values using linear regression + gradient boosting')

    out_original = (data[['Receipt_Count']].groupby(pd.to_datetime(data.index).to_period('M')).sum())
    out = (data_new[['Receipt_Count_Predicted']].groupby(pd.to_datetime(data_new.index).to_period('M')).sum())

    return fig, out_original, out
