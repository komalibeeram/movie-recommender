import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
df2 = pd.read_csv("item.csv")
#print(df.head(5))
#print(df2.head(2))
df1=df['item_id'].value_counts().rename_axis('item_id').reset_index(name='vote_count')
df1['vote_average']=df.groupby('item_id')['rating'].mean()
df1['vote_average']=df1['vote_average'].replace(to_replace = np.nan, value= 4.358491)
#print(df1.head(5))
C=df1['vote_average'].mean()
print("Value of C: ",C)
m=df1['vote_count'].quantile(0.9)
print("Value of m: ",m)
movies = df1.loc[df1['vote_count']>=m].copy()
#print(movies)
def weighted_rating(x,m=m,C=C):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m) * R)+(m/(m+v) * C)
movies['score'] = movies.apply(weighted_rating,axis=1)
movies = movies.sort_values('score',ascending=False)
#print(movies[['item_id','vote_count','vote_average','score']].head(10))
df3=pd.merge(movies,df2,left_on=['item_id'],right_on=['movie_id'],how='left')
print(df3[['item_id','vote_count','vote_average','score','movie_title']].head(10))
