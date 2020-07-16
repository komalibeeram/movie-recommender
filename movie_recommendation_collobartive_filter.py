import pandas as pd
import numpy as np
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import SVD
from surprise import NMF
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from collections import defaultdict
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.head(5))

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

b = []
# Iterate over all algorithms
for algorithm in [SVD(), NMF(), KNNBasic()]:
    # Perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE'], cv=3, verbose=False)
    
    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]],index=['Algorithm']))
    b.append(tmp)

df1 = pd.DataFrame(b).set_index('Algorithm').sort_values('test_rmse')
print(df1)
#Tuning parameters using gridsearchcv
param_grid = {'n_factors': [25, 30, 35, 40], 'n_epochs': [15, 20, 25], 'lr_all': [0.001, 0.003, 0.005, 0.008],
              'reg_all': [0.08, 0.1, 0.15]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
gs.fit(data)
algo = gs.best_estimator['rmse']
print(gs.best_score['rmse'])
print(gs.best_params['rmse'])

#Assigning values
t = gs.best_params
factors = t['rmse']['n_factors']
epochs = t['rmse']['n_epochs']
lr_value = t['rmse']['lr_all']
reg_value = t['rmse']['reg_all']
#training and splittin data
trainset, testset = train_test_split(data, test_size=0.25)
algo = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr_value, reg_all=reg_value)
predictions = algo.fit(trainset).test(testset)
print(accuracy.rmse(predictions))

def get_Iu(uid):#returns no. of items rated by the user
    try:
        return len(trainset.ur[trainset.to_inner_uid(uid)])
    except ValueError: # user was not part of the trainset
        return 0
    
def get_Ui(iid):#returns number of users that have rated the item
    try: 
        return len(trainset.ir[trainset.to_inner_iid(iid)])
    except ValueError:
        return 0

    
df_predictions = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
df_predictions['Iu'] = df_predictions.uid.apply(get_Iu)
df_predictions['Ui'] = df_predictions.iid.apply(get_Ui)
df_predictions['err'] = abs(df_predictions.est - df_predictions.rui)

final = []

for threshold in np.arange(0, 5.5, 0.5):
  tp=0
  fn=0
  fp=0
  tn=0
  temp = []

  for user_id, _, true_rating, est_rating, _ in predictions:
    if(true_r>=threshold):
      if(est_rating>=threshold):
        tp = tp+1
      else:
        fn = fn+1
    else:
      if(est_rating>=threshold):
        fp = fp+1
      else:
        tn = tn+1   

    if tp == 0:
      precision = 0
      recall = 0
      f1 = 0
    else:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * (precision * recall) / (precision + recall)  

  temp = [threshold, tp,fp,tn ,fn, precision, recall, f1]
  final.append(temp)

results = pd.DataFrame(final)
results.rename(columns={0:'threshold', 1:'tp', 2: 'fp', 3: 'tn', 4:'fn', 5: 'Precision', 6:'Recall', 7:'F1'}, inplace=True)
print(results)

def precision_recall_at_k(predictions, k, threshold):
    #mapping the predictions to each user.
    user_est_true = defaultdict(list)
    for user_id, _, true_rating, est_rating, _ in predictions:
        user_est_true[uid].append((est_rating, true_rating))

    precisions = dict()
    recalls = dict()
    for user_id, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_rating >= threshold) for (_, true_rating) in user_ratings)
        n_rec_k = sum((est_rating >= threshold) for (est_rating, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(((true_rating >= threshold) and (est_rating >= threshold))
                              for (est_rating, true_rating) in user_ratings[:k])
        precisions[user_id] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
        recalls[user_id] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precisions, recalls

results=[]
for i in range(2, 11):
    precisions, recalls = precision_recall_at_k(predictions, k=i, threshold=2.5)
    # Precision and recall averaged over all users
    prec = sum(prec for prec in precisions.values()) / len(precisions)
    rec = sum(rec for rec in recalls.values()) / len(recalls)
    results.append({'K': i, 'Precision': prec, 'Recall': rec})
    
print(results)

Rec=[]
Precision=[]
Recall=[]
for i in range(0,9):
    Rec.append(results[i]['K'])
    Precision.append(results[i]['Precision'])
    Recall.append(results[i]['Recall'])

plt.plot(Rec, Precision)
plt.xlabel('# of Recommendations')
plt.ylabel('Precision')
plt2 = plt.twinx()
plt2.plot(Rec, Recall, 'r')
plt.ylabel('Recall')
for tl in plt2.get_yticklabels():
    tl.set_color('r')
plt.show()

trainset = data.build_full_trainset()   #Build on entire data set
algo = SVD(n_factors=factors, n_epochs=epochs, lr_all=lr_value, reg_all=reg_value)
algo.fit(trainset)
# Predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
#Predicting the ratings for testset
predictions = algo.test(testset)

def get_all_predictions(predictions):
    #map the predictions to each user.
    top_n = defaultdict(list)    
    for user_id, item_id, true_rating, est_rating, _ in predictions:
        top_n[user_id].append((item_id, est_rating))
    #sort the predictions for each user
    for user_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
    return top_n

all_pred=get_all_predictions(predictions)
n = 7
for user_id, user_ratings in all_pred.items():
    user_ratings.sort(key=lambda x: x[1], reverse=True)
    all_pred[user_id] = user_ratings[:n]
tmp = pd.DataFrame.from_dict(all_pred)
tmp_transpose = tmp.transpose()
def get_predictions(user_id):
    results = tmp_transpose.loc[user_id]
    return results
user_id=1
results = get_predictions(user_id)
print(results)
recommended_movie_ids=[]
for x in range(0, n):
    recommended_movie_ids.append(results[x][0])
print(recommended_movie_ids)
movies = pd.read_csv('item.csv')
movies.head()
recommended_movies = movies[movies['movie_id'].isin(recommended_movie_ids)]
print(recommended_movies)
