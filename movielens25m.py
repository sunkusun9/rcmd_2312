import os
import pickle
import pandas as pd
import numpy as np

def load_dataset(path):
    # 영화 메타데이터를 가져옵니다. (한국어 버젼)
    with open(os.path.join(path, 'tmdb_movie_info_kr.pkl'), 'rb') as f:
        dic_movie_info_kr = pickle.load(f)

    # OpenAI의 Embedding API를 통해 구한 영화 타이틀과 줄거리의 Embedding 데이터를 불러 옵니다.
    with open(os.path.join(path, 'tmdb_movie_emb.pkl'), 'rb') as f:
        movie_info_emb = pickle.load(f)

    # 평점을 불러옵니다.
    dtype = {'rating': 'float16'}
    df_ratings = pd.read_csv(os.path.join(path, 'ratings.csv'), dtype=dtype)
    # 일자를 timestamp 형식(Integer)에서 일자형식으로 바꿉니다.
    df_ratings['date'] = df_ratings.pop('timestamp').pipe(lambda x: pd.to_datetime(x, unit='s'))

    # 영화의 간략 버젼의 메타데이터를 불러옵니다.
    df_movie = pd.read_csv(os.path.join(path, 'movies.csv'), index_col='movieId')

    # 영화 출시일을 영화 메타데이터에서 가져옵니다.
    df_movie['release_date'] = df_movie.index.map(
        lambda x: dic_movie_info_kr.get(x, {'release_date': None}).get('release_date', None)
    ).values

    # 결측일 경우 평점 데이터에서 가장 먼저 출현한 시점으로 잡습니다.
    s_date_fillna = df_ratings.loc[
        df_ratings['movieId'].isin(df_movie.loc[df_movie['release_date'].isna()].index)
    ].groupby('movieId')['date'].min()
    df_movie.loc[df_movie.index.isin(s_date_fillna.index), 'release_date'] = pd.to_datetime(s_date_fillna)
    df_movie['release_date'] = pd.to_datetime(df_movie['release_date'] )

    # 영화 출시 시점과 평가 시점과 차(일자 단위)
    df_ratings['days'] = (df_ratings['date'] - df_ratings['movieId'].map(df_movie['release_date'])).dt.days.clip(0)

    # 영화 당 여러 개의 genre(장르)가 부여됩니다(movie:genre=1:N 관계). 이를 위한 처리를 합니다. (movieId, genre 리스트)인 튜플로 된 리스트를 만듭니다.
    l_genre = [(k, [i['name'] for i in v['genres']]) for k, v in dic_movie_info_kr.items()]
    # movieId, genres 리스 컬럼으로된 DataFrame을 만듭니다
    df_genre = pd.DataFrame(l_genre, columns=['movieId', 'genres']).set_index('movieId')
    # movieId, genre로 된 DataFrame을 만듭니다.
    df_genre = df_genre['genres'].explode().dropna()
    # genre를 1 기준(1-base)의 인덱스로 변환하기 위한 매핑을 만듭니다.
    s_genre_map = pd.Series(np.arange(1, df_genre.nunique() + 1), index=df_genre.unique())
    # 처리 효율성을 위해 영화의 genre 정보를 genre에 대한 1-기준(1-base) 인덱스으로 변환합니다.
    s_genre = df_genre.map(s_genre_map).pipe(lambda x: x.groupby(level=0).agg(list))
    del l_genre, df_genre

    # 영화의 시리즈 정보를 정리합니다.(movie:series=1:1 관계)
    l_series = [
        (k,  v['belongs_to_collection']['name']) 
        for k, v in dic_movie_info_kr.items() if v['belongs_to_collection'] is not None
    ]
    # 영화에 부여된 시리즈  정보를 pd.Series 형태로 만듭니다.
    s_series = pd.DataFrame(l_series, columns=['movieId', 'collection']).set_index('movieId')['collection']
    # 영화의 시리즈 정보를 1-base 인덱스로 변환하기 위한 매핑을 합니다.
    s_collection = pd.Series(np.arange(1, s_series.nunique() + 1), index = s_series.unique())
    # 영화의 시리즈 정보를 1-base 인덱스로 변환합니다. 
    s_series = s_series.map(s_collection)
    del l_series

    # 추출한 영화 메타데이터를 정리합니다.
    df_movieinfo = pd.concat([s_genre, s_series], axis=1)

    # NA를 [0] List 타입으로 설정이 안 됩니다.
    df_movieinfo.loc[df_movieinfo['genres'].isna(), 'genres'] = \
        df_movieinfo.loc[df_movieinfo['genres'].isna()].pipe(lambda x: pd.Series([[0]] * len(x), index = x.index))
    df_movieinfo['collection'] = df_movieinfo['collection'].fillna(0).astype(int)

    df_movieinfo['ov_emb'] = df_movieinfo.index.to_series().map(movie_info_emb)
    avg_ov_emb = np.mean(
        np.vstack(
            df_movieinfo['ov_emb']
        ), axis=0
    )
    df_movieinfo = pd.concat([
        pd.DataFrame({'genres': [[0]], 'collection': [0], 'ov_emb': [avg_ov_emb]}),
        df_movieinfo
    ], axis=0)


    return {
        'movie_info_kr': dic_movie_info_kr,
        'movie': df_movie,
        'genre': s_genre_map,
        'series': s_collection,
        'ratings': df_ratings,
        "movieinfo": df_movieinfo
    }