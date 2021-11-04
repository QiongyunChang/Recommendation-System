"""
NM6104054 張瓊云
(1) User‐based Collaborative Filtering (U‐CF)
(2) Item‐based Collaborative Filtering (I‐CF)

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import Latent_Matrix as LM


#  參考網站:
# https://www.kaggle.com/fuzzywizard/rec-sys-collaborative-filtering-dl-techniques#4)-Matrix-Factorization-using-Deep-Learning-(Keras)


# 讀取 ratings.data文件
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ratings.data', sep='\t', names=header)

# 計算唯一用戶和電影的數量
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print('Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items))
# 分割資料
train_data, test_data = train_test_split(df, test_size=0.25)


# 協同過濾算法
# 第一步是創建 uesr-item 矩陣，此處需創建訓練和測試兩個 UI矩陣
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]

# 相似度計算 - Cosine
def cosine_similarity(ratings,kind='user',epsilon=1e-9):
    if kind=='user':
        sim=ratings.dot(ratings.T)+epsilon
    elif kind=='movie':
        sim=ratings.T.dot(ratings)+epsilon
    norms=np.array(np.sqrt(np.diagonal(sim)))
    return (sim/norms/norms.T)
# from sklearn.metrics.pairwise import cosine_similarity
# predictions = P.dot(Q.T)
# mask = np.zeros_like(ratings)
# mask[ratings.nonzero()] = 1

user_similarity=cosine_similarity(train_data_matrix,kind='user')
movie_similarity=cosine_similarity(train_data_matrix,kind='movie')
user_similarity_pcc =  np.corrcoef(train_data_matrix)
# 處理 NAN 數值問題
# ratings = train_data_matrix.copy()
# mask = np.zeros_like(ratings)
# mask[ratings.nonzero()] = 1
# np.round(predictions * mask, 2)
user_similarity_pcc[np.isnan(user_similarity_pcc)] = 0
movie_similarity_pcc =np.corrcoef(train_data_matrix.T)
movie_similarity_pcc[np.isnan(movie_similarity_pcc)] = 0
#
# print(movie_similarity_pcc)
# print(user_similarity_pcc)


# 利用前 K 個相近喜好來預測
def predict_topk(ratings,similarity,kind='user',k=40):
    pred=np.zeros(ratings.shape)
    if kind=='user':
        for i in range(ratings.shape[0]):
            top_k_users=np.argsort(similarity[:,i])[:-k-1:-1]
            pred[i,:]=similarity[i,[top_k_users]].dot(ratings[top_k_users,:])
            pred[i,:]/=np.sum(np.abs(similarity[i,[top_k_users]]))
    if kind=='movie':
        for j in range(ratings.shape[1]):
            top_k_movies=np.argsort(similarity[:,j])[:-k-1:-1]
            pred[:,j]=ratings[:,top_k_movies].dot(similarity[top_k_movies,j].T)
            pred[:,j]/=np.sum(np.abs(similarity[top_k_movies,j]))
    return pred

#  進行 Evaluation
def get_mse(pred,actual):
    pred=pred[actual.nonzero()].flatten()
    actual=actual[actual.nonzero()].flatten()
    return mean_squared_error(pred,actual)

k_array = [3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]

user_pcc_mse = []
user_cos_mse = []
item_cos_mse = []
item_pcc_mse = []

for k in k_array:
    user_pred_cos = predict_topk(train_data_matrix, user_similarity, kind='user', k=k)
    item_pred_cos = predict_topk(train_data_matrix, movie_similarity, kind='movie', k=k)
    user_pred_pcc = predict_topk(train_data_matrix, user_similarity_pcc, kind='user', k=k)
    item_pred_pcc = predict_topk(train_data_matrix, movie_similarity_pcc, kind='movie', k=k)

    user_cos_mse += [get_mse(user_pred_cos, test_data_matrix)]
    item_cos_mse += [get_mse(item_pred_cos, test_data_matrix)]
    user_pcc_mse += [get_mse(user_pred_pcc, train_data_matrix)]
    item_pcc_mse += [get_mse(item_pred_pcc, train_data_matrix)]
print(user_cos_mse )
print(item_cos_mse )


 # 畫圖
sns.set()
pal = sns.color_palette("Set2", 4)
plt.figure(figsize=(8, 8))
plt.plot(k_array, item_pcc_mse, c=pal[0], label='I-CF-PCC',  linewidth=3)
plt.plot(k_array, user_cos_mse, c=pal[1], label='I-CF-COS', linewidth=3)
plt.plot(k_array, user_pcc_mse, c=pal[2], label='U-CF-PCC', linewidth=3)
plt.plot(k_array, item_cos_mse, c=pal[3], label='U-CF-COS', linewidth=3)
plt.legend(loc='best', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('k', fontsize=30)
plt.ylabel('MSE', fontsize=30)
plt.show()






"""
Compare the RMSE　
"""



user_pred_cos = predict_topk(train_data_matrix, user_similarity, kind='user', k=6)
item_pred_cos = predict_topk(train_data_matrix, movie_similarity, kind='movie', k=6)
user_pred_pcc = predict_topk(train_data_matrix, user_similarity_pcc, kind='user', k=6)
item_pred_pcc = predict_topk(train_data_matrix, movie_similarity_pcc, kind='movie', k=6)

u_cf_cos = [get_mse(user_pred_cos, test_data_matrix)]
i_cf_cos = [get_mse(item_pred_cos, test_data_matrix)]
u_cf_pcc = [get_mse(user_pred_pcc, train_data_matrix)]
i_cf_pcc = [get_mse(item_pred_pcc, train_data_matrix)]
mf_bias_l = LM.MF(train_data_matrix, K=6, alpha=0.1, beta=0.1, iterations=10, type = 'bias')
training_process_bias,mf_bias_1 = LM.mf_bias_l.train()
mf_nobias_l = LM.MF(train_data_matrix, K=6, alpha=0.1, beta=0.1, iterations=10, type = 'nonbias')
training_process_bias,mf_nobias_1 = LM.mf_nobias_l.train()
mf_bias = mf_bias_l[-1]
mf_nobias = mf_nobias_l[-1]

method = ['Default parameter', 'Tuned parameter']
u_cf_cos_label = [u_cf_cos, u_cf_cos]
i_cf_cos_label = [i_cf_cos, i_cf_cos]
u_cf_pcc_label = [u_cf_pcc,u_cf_pcc]
i_cf_pcc_label = [i_cf_pcc,i_cf_pcc]
mf_bias_label = [mf_bias,mf_bias]
mf_nobias_label = [mf_nobias,mf_nobias]

x = np.arange(len(method))
width = 0.3
plt.bar(x, u_cf_cos_label, width, color='green', label='Math')
plt.bar(x + width, i_cf_cos_label, width, color='blue', label='History')
plt.bar(x+ 2*width, u_cf_pcc_label, width, color='green', label='Math')
plt.bar(x + 3*width, i_cf_pcc_label, width, color='blue', label='History')
plt.bar(x + 4*width, mf_bias_label, width, color='green', label='Math')
plt.bar(x + 5*width, mf_nobias_label, width, color='blue', label='History')
plt.xticks(x + 5*width / 6, method)
plt.ylabel('RMSE')
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.show()