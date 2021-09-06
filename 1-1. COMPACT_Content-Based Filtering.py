import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('./data/TMDB_5000_Movie_Dataset/tmdb_5000_movies.csv')

percentile = 0.6

def text_to_lObj(df, col):
    return df[col].apply(literal_eval).apply(lambda x : [y['name'] for y in x])

def extract(keywords):
    feature = data[keywords]
    feature['genres'] = text_to_lObj(feature, 'genres')
    feature['keywords'] = text_to_lObj(feature, 'keywords')
    feature['genres_literal'] = feature['genres'].apply(lambda x: (' ').join(x))
    return feature

def calc_sim(matrix):
    genre_sim = cosine_similarity(matrix, matrix)
    return genre_sim.argsort()[:, ::-1][:1]

def find_sim_movie(df, sorted_ind, title_name, top_n = 10):
    title_movie = df[df['title'] == title_name]
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    similar_indexes = similar_indexes.reshape(-1)

    return df.iloc[similar_indexes]

def weighted_vote_average(record):
    c = record['vote_average'].mean()
    m = record['vote_count'].quantile(percentile)
    v = record['vote_count']
    r = record['vote_average']

    return ((v/(v+m)) * r) + ((m/(v+m)) * c)

def find_sim_movie(df, sorted_ind, index, top_n = 10):
    title_movie = df[df['title'].index.values == index]
    title_index = title_movie.index.values
    
    similar_indexes = sorted_ind[title_index, :(top_n * 2)]
    similar_indexes = similar_indexes.reshape(-1)
    
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

def main(index, n):
    col_names = ['id','title','genres','vote_average','vote_count','popularity','keywords','overview']
    feature = extract(col_names)
    
    count_vect = CountVectorizer(min_df = 0, ngram_range=(1,2))
    genre_mat = count_vect.fit_transform(feature['genres_literal'])
    genre_sim_sorted_ind = calc_sim(genre_mat)    

    feature['weighted_vote'] = feature.apply(weighted_vote_average, axis=1)
    similar_movies = find_sim_movie(feature, genre_sim_sorted_ind, index, n)

    return similar_movies[['title', 'vote_average', 'weighted_vote']]

if __name__ == '__main__':
    main(10, 10)