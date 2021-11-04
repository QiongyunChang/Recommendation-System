#   Latent Factor Model (Matrix Factorization)
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


# 參考 https://github.com/yanneta/pytorch-tutorials/blob/master/collaborative-filtering-nn.ipynb

# read the  ratings file
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ratings.data', sep='\t', names=header)
# print(df)

# 計算唯一用戶和電影的數量
num_users = df.user_id.unique().shape[0]
num_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(num_users) + ' | Number of movies = ' + str(num_items))

train_data, test_data = train_test_split(df, test_size=0.25)

train_data_matrix = np.zeros((num_users, num_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((num_users, num_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]


class MF():

    def __init__(self, R, K, alpha, beta, iterations, type='bias'):
        """
        Perform matrix factorization to predict empty entries in a matrix.
        - R (ndarray)   : user-item rating matrix
        - K (int)       : number of latent dimensions
        - alpha (float) : learning rate
        - beta (float)  : regularization parameter
        """

        self.R = R
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.type = 'bias'

    def train(self):
        # Initialize user and item latent feature matrice
        self.P = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K)).astype(np.float64)
        self.Q = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K)).astype(np.float64)

        # Initialize the biases
        self.b_u = np.zeros(self.num_users).astype(np.float64)
        self.b_i = np.zeros(self.num_items).astype(np.float64)
        self.b = np.mean(self.R[np.where(self.R != 0)]).astype(np.float64)
        # Create a list of training samples
        self.samples = [
            (i, j, self.R[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.R[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        test_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            # print(self.b_u)
            # print("YYYY")
            mse = self.mse()
            evalution_rmse = self.eval()
            # print(mse)
            # print(i,'1')
            training_process.append(mse)
            test_process.append(evalution_rmse)
            # print(test_process)

            if (i + 1) % 10 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, mse))

        return training_process, test_process

    def mse(self):
        """
        A function to compute the total mean square error
        """
        xs, ys = self.R.nonzero()
        predicted = self.full_matrix()
        error = 0
        for x, y in zip(xs, ys):
            # print(predicted[x, y], self.R[x, y] )
            error += pow(self.R[x, y] - predicted[x, y], 2)
        return np.sqrt(error)

    def sgd(self):
        """
        Perform stochastic graident descent
        """
        for i, j, r in self.samples:
            # Computer prediction and error
            if (self.type=='bias'):
                prediction = self.get_rating_bias(i, j)
            elif(self.type=='nonbias') :
                prediction = self.get_rating(i, j)
            # print(i, j, r,prediction)
            e = (r - prediction)

            # Update biases
            self.b_u[i] =self.b_u[i]+ self.alpha * (e - self.beta * self.b_u[i])
            self.b_i[j] = self.b_i[j]  + self.alpha * (e - self.beta * self.b_i[j])

            # Create copy of row of P since we need to update it but use older values for update on Q
            P_i = self.P[i, :][:]

            # Update user and item latent feature matrices
            # print(self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :]))
            # print(self.P[i, :])
            self.P[i, :] =self.P[i, :] + self.alpha * (e * self.Q[j, :] - self.beta * self.P[i, :])
            # print(self.P[i, :],"&&&&&&")
            self.Q[j, :] = self.Q[j, :] + self.alpha * (e * P_i - self.beta * self.Q[j, :])
            # print(self.Q[j, :])


    def get_rating_bias(self, i, j):
        """
        Get the predicted rating of user i and item j with adding the bias
        """
        prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def get_rating(self, i, j):
        """
        Get the predicted rating of user i and item j
        """
        prediction = self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    def full_matrix(self):
        """
        Computer the full matrix using the resultant biases, P and Q
        """
        # print(self.b,np.shape(self.b))
        # print(self.b_u[:, np.newaxis])
        # print(self.b_i[np.newaxis:, ],"TTTT")
        # print(self.P.dot(self.Q.T))
        if (self.type=='bias'):
            return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis:, ] + self.P.dot(self.Q.T)
        elif (self.type=='nonbias'):
            return self.P.dot(self.Q.T)

    # for testing error
    from sklearn.metrics import mean_squared_error

    def eval(self):
        # print(self.full_matrix().shape)
        test_error = np.sqrt(((self.full_matrix() -  test_data_matrix) ** 2).mean())
        # test_error = mean_squared_error(self.full_matrix(), test_data, squared=False)
        return test_error
#
#
# # Perform training and obtain the user and item matrices
mf = MF(train_data_matrix, K=100, alpha=0.1, beta=0.1, iterations=20, type = 'bias')
mf2 = MF(train_data_matrix, K=100, alpha=0.1, beta=0.1, iterations=20, type = 'nonbias')

training_process_bias,test_process_bias = mf.train()
# print(training_process_bias,'+++++++++')
# print(test_process_bias,'_______')
# test_process_bias = mf.eval()
training_process,test_process  = mf2.train()
# test_process = mf2.eval()
#


import matplotlib.pyplot as plt
 # draw the plot with iteration
plt.figure(figsize=(12, 4))
colors=['orange', 'blue', 'green','red']
plt.gca().set_prop_cycle(color=colors)
plt.plot(range(len(training_process_bias)), training_process_bias,label='MF-bias(training)')
plt.plot(range(len(test_process_bias)), test_process_bias, label = 'MF-bias(testing)')
plt.plot(range(len(training_process)), training_process, label = 'MF-nonbias(training)')
plt.plot(range(len(test_process)), test_process, label = 'MF-nonbias(testing)')
plt.legend()
plt.xlabel("Iterations")
plt.ylabel(" RMSE")
plt.grid(axis="y")
plt.show()

#  Deal with the K latent
k_latent = np.arange(5,105,5)
training_process_bias_k =[]
test_process_bias_k=[]
training_process_k=[]
test_process_k = []

for i in k_latent:
    mf_k = MF(train_data_matrix, K=i, alpha=0.1, beta=0.1, iterations=10, type = 'bias')
    mf_k2 = MF(train_data_matrix, K=i, alpha=0.1, beta=0.1, iterations=10, type = 'nonbias')
    training_process_bias_la, test_process_bias_la = mf_k.train()
    training_process_la, test_process_la = mf_k2.train()
    training_process_bias_k.append(training_process_bias_la[-1])
    test_process_bias_k.append(test_process_bias_la[-1])
    training_process_k.append(training_process_la[-1])
    test_process_k.append(test_process_la[-1])

# print(training_process_bias_k)
# print(test_process_bias_k)
# print(training_process_k)
# print(test_process_k)

plt.figure(figsize=(12, 4))
colors=['orange', 'blue', 'green','red']
plt.gca().set_prop_cycle(color=colors)
plt.plot(k_latent, training_process_bias_k,label='MF-bias(training)')
plt.plot(k_latent, test_process_bias_k, label = 'MF-bias(testing)')
plt.plot(k_latent, training_process_k, label = 'MF-nonbias(training)')
plt.plot(k_latent, test_process_k, label = 'MF-nonbias(testing)')
plt.legend()
plt.xlabel("K-user")
plt.ylabel(" RMSE")
plt.grid(axis="y")
plt.show()



