import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, hamming_loss
from skmultilearn.model_selection import iterative_train_test_split
import pickle

def clean_data(df):
    # Unpack values from the dictionaries to lists for columns for the better usability: 
    # genres, production_companies, production_countries and spoken_languages
    # Fill missing values with Unknown
    print("genres, production_companies, production_countries, spoken_languages")
    df['genres'] = df['genres'].apply(lambda x: ','.join([d['name'] for d in x]))
    df['production_companies'] = df['production_companies'].apply(lambda x: ','.join([d['name'] for d in x]))
    df['production_countries'] = df['production_countries'].apply(lambda x: ','.join([d['iso_3166_1'] for d in x]))
    df['spoken_languages'] = df['spoken_languages'].apply(lambda x: ','.join([d['iso_639_1'] for d in x]))
    df['production_companies'] = df['production_companies'].replace('', 'Unknown')
    df['production_countries'] = df['production_countries'].replace('', 'Unknown')
    df['spoken_languages'] = df['spoken_languages'].replace('', 'Unknown')
    df['genres'] = df['genres'].replace('', 'Unknown')

    # Creating a dictionary to map non-standard ratings to standard MPA ratings
    print("Fix Ratings")
    rating_map = {
        'Not Rated': 'Unrated',
        'N/A': 'Unrated',
        'Approved': 'Other',
        'Passed': 'Other',
        'GP': 'PG',  # Assuming GP (General Public) is similar to PG
        'M': 'PG', # "M" was renamed to "GP" so also PG
        'X': 'NC-17',
        'M/PG': 'PG',
        'TV-PG': 'PG',
        'TV-MA': 'R',
        'TV-14': 'PG-13',
        '18+': 'NC-17',
        '16+': 'R',
        '13+': 'PG-13',
        'TV-G': 'G',
        'TV-Y7': 'G'
    }
    # Applying the mapping to the 'Rated' column
    df['Rated'] = df['Rated'].replace(rating_map)

    print('excluding Documentary and TV Movie')
    # exclude 'Documentary' and 'TV Movie' and take only movie category
    df = df[df['Type'].str.lower() == 'movie']
    df = df[df['genres'].apply(lambda x: 'TV Movie' not in x and 'Documentary' not in x )]
    df = df[df['genres'].apply(lambda x: len(x) > 0)]

    # awards
    print('unpacking awards')
    import re

    pattern = {'oscar_won': re.compile(r'won (\d+) oscar', re.IGNORECASE),
            'oscar_nominated': re.compile(r'nominated for (\d+) oscar', re.IGNORECASE),
            'bafta_won': re.compile(r'won (\d+) bafta', re.IGNORECASE),
            'bafta_nominated': re.compile(r'nominated for (\d+) bafta', re.IGNORECASE),
            'awards_won': re.compile(r'(\d+)\s*win', re.IGNORECASE),
            'awards_nominated': re.compile(r'(\d+)\s*nomination', re.IGNORECASE)}
    df['Awards'] = df['Awards'].str.replace('n/a', '')
    for k, v in pattern.items():
        df[k] = df['Awards'].str.extract(v).fillna(0).astype(int)

    print('unpacking ratings')
    df['imdbRating'] = pd.to_numeric(df['imdbRating'], errors='coerce')
    df['imdbRating'] = df['imdbRating'] * 10
    df['vote_average'] = df['vote_average'] * 10
    df['imdbVotes'] = df['imdbVotes'].str.replace(',', '')
    df['imdbVotes'] = pd.to_numeric(df['imdbVotes'], errors='coerce')
    
    print('adjust inflation')
    cpi_data = pd.read_csv('US_inflation_rates.csv', names=['date', 'CPI'], skiprows=1)  

    # Convert the 'date' column to datetime format and extract the year and month
    cpi_data['date'] = pd.to_datetime(cpi_data['date'], format='%Y-%m-%d')
    cpi_data['year'] = cpi_data['date'].dt.year
    cpi_data['month'] = cpi_data['date'].dt.month

    # get relase year and release month
    df['release_month'] = pd.to_datetime(df['release_date']).dt.month
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year

    # Merge the datasets on year and month
    merged_data = pd.merge(df, cpi_data, left_on=['release_year', 'release_month'], right_on=['year', 'month'], how='left')

    # filter movies with budget or revenue  with less than 1000$
    merged_data = merged_data[(merged_data['budget'] >= 1000) & (merged_data['revenue'] >= 1000) & (merged_data['budget'] != merged_data['revenue'])]

    # Find the most recent year and month in the dataset
    max_year = cpi_data['year'].max()
    max_month = cpi_data[cpi_data['year'] == max_year]['month'].max()

    # Get the CPI value for the target year and month
    target_cpi = cpi_data[(cpi_data['year'] == max_year) & (cpi_data['month'] == max_month)]['CPI'].values[0]

    # Calculate the adjustment factor
    merged_data['adjustment_factor'] = target_cpi / merged_data['CPI']

    # Adjust the budget and revenue columns
    merged_data['adjusted_budget'] = merged_data['budget'] * merged_data['adjustment_factor']
    merged_data['adjusted_revenue'] = merged_data['revenue'] * merged_data['adjustment_factor']

    # Set the float format to display the entire number
    pd.options.display.float_format = '{:.2f}'.format

    df = merged_data[merged_data['release_year'] >= 1947].drop(['year', 'month', 'date', 'CPI', 'adjustment_factor', 'budget', 'revenue'], axis=1)
    df = df.dropna(subset=['adjusted_budget', 'adjusted_revenue']) # ir file is limited to 2023-06-01

    print("unpack crew and cast")
    df = df[df['crew'].notnull()]
    df = df[df['cast'].notnull()]

    def clean_cast_members(members):
        # Retain only 'name' and 'popularity' keys for each member
        return [{'name': member['name'].lower(), 'popularity': member['popularity']} for member in members]

    def clean_crew_members(members):
        # Retain only 'name', 'popularity', and 'job' keys for each member
        return [{'name': member['name'].lower(), 'popularity': member['popularity'], 'job': member['job'].lower()} for member in members]

    df['cast'] = df['cast'].apply(clean_cast_members)
    df['crew'] = df['crew'].apply(clean_crew_members)
    df = df[df['cast'].apply(lambda x: len(x) > 0)]
    df = df[df['crew'].apply(lambda x: len(x) > 0)]


    print("Fix columns")
    dataset_df = df[["original_language" , 'spoken_languages', 'genres', 'release_month',
                        'production_companies',  'production_countries', 'runtime', 'Rated', 
                           'vote_average', 'vote_count', 'cast', 'crew', 'belongs_to_collection.name',
                        'oscar_won', 'oscar_nominated', 'bafta_won', 'bafta_nominated', 'awards_won', 'awards_nominated',
                        'imdbRating', 'imdbVotes', 'adjusted_budget', 'adjusted_revenue']].copy()
    # Standardize column names
    dataset_df.columns = dataset_df.columns.str.lower().str.replace(' ', '_')


    dataset_df.rename(columns={"popularity": "tmdb_popularity", "vote_average": "tmdb_rating",
                        "vote_count": "tmdb_vote_count", 
                        "imdbrating": "imdb_rating", 'imdbvotes': 'imdb_votes'}, inplace=True)

    # lower all values in categorical columns
    categorical_vars = list(dataset_df.dtypes[dataset_df.dtypes == 'object'].index)
    categorical_vars.remove('crew')
    categorical_vars.remove('cast')
    for c in categorical_vars:
        dataset_df[c] = dataset_df[c].str.lower()
    
    return dataset_df

def features_extraction(df):
    print('ROI')
    df['adjusted_ROI'] = ((df['adjusted_revenue'] - df['adjusted_budget']) / df['adjusted_budget']) * 100

    print('Compute ROI labels')
    def categorize_roi(roi):
        if roi < 300:
            return 'flop'
        else:
            return 'hit'

    df['ROI_category'] = df['adjusted_ROI'].apply(categorize_roi)

    label_mapping = {
        'flop': 0,
        'hit': 1
    }
    df['numerical_ROI_category'] = df['ROI_category'].map(label_mapping)

    print('compute weighted rating')
    m = df['tmdb_vote_count'].quantile(0.70)  # minimum votes required to be listed
    C = df['tmdb_rating'].mean()  # mean rating across the whole dataset

    # Function to compute weighted rating
    def weighted_rating(x, m=m, C=C):
        v = x['tmdb_vote_count']
        R = x['tmdb_rating']
        return (v / (v + m) * R) + (m / (m + v) * C)

    df['weighted_tmdb_rating'] = df.apply(weighted_rating, axis=1)

    m = df['imdb_votes'].quantile(0.70)  # minimum votes required to be listed
    C = df['imdb_rating'].mean()  # mean rating across the whole dataset

    # Function to compute weighted rating
    def weighted_rating(x, m=m, C=C):
        v = x['imdb_votes']
        R = x['imdb_rating']
        return (v / (v + m) * R) + (m / (m + v) * C)

    df['weighted_imdb_rating'] = df.apply(weighted_rating, axis=1)

    df['average_rating'] = np.where(df['weighted_imdb_rating'].isnull(), 
                                    df['weighted_tmdb_rating'], 
                                    (df['weighted_tmdb_rating'] + df['weighted_imdb_rating']) / 2)
   
    print('compute rating labels')
    df['rating_category'] = 'flop' 
    top_25_percentile_threshold = df['average_rating'].quantile(0.75)  
    df.loc[df['average_rating'] >= top_25_percentile_threshold, 'rating_category'] = 'hit'  

    label_mapping = {
        'flop': 0,
        'hit': 1
    }
    
    df['numerical_rating_category'] = df['rating_category'].map(label_mapping)
    
    print('compute awards labels')
    df['award_points'] = (df['oscar_won'] * 5 + df['oscar_nominated'] * 3 +
                      df['bafta_won'] * 4 + df['bafta_nominated'] * 2 +
                      (df['awards_won'] - df['oscar_won'] - df['bafta_won']) * 3 + 
                      (df['awards_nominated'] - df['oscar_nominated'] - df['bafta_nominated']) * 1)

    # Determine the 75th percentile value
    threshold = df['award_points'].quantile(0.75)

    df['award_category'] = ['hit' if x >= threshold else 'flop' for x in df['award_points']]

    label_mapping = {
        'flop': 0,
        'hit': 1
    }
    df['numerical_award_category'] = df['award_category'].map(label_mapping)

    print('features')
    df['log_adjusted_budget'] = np.log(df['adjusted_budget'])
    df['collection'] = df['belongs_to_collection.name'].notna()
    df['is_english'] = df['original_language'].apply(lambda x: 1 if x=='en' else 0)
    
    def get_season(month):
        if month in [1, 2]:
            return 'winter'
        elif month == 12:
            return 'holidays'
        elif month in [3, 4]:
            return 'spring'
        elif month in [5, 6, 7]:
            return 'summer'
        else:
            return 'fall'

    df['seasons'] = df['release_month'].apply(get_season)

    seasons_dummies = df['seasons'].str.get_dummies(sep=',')
    df = pd.concat([df, seasons_dummies], axis=1)

    def replace_genre(genre_string):
        genre_string = genre_string.replace('adventure', 'moderate performing')
        genre_string = genre_string.replace('comedy','moderate performing')
        genre_string = genre_string.replace('fantasy', 'moderate performing')
        genre_string = genre_string.replace('romance', 'moderate performing')
        genre_string = genre_string.replace('science fiction', 'moderate performing')
        genre_string = genre_string.replace('thriller', 'moderate performing')
        genre_string = genre_string.replace('crime', 'others')
        genre_string = genre_string.replace('western', 'others')
        genre_string = genre_string.replace('unknown', 'others')
        genre_string = genre_string.replace('music', 'others')
        genre_string = genre_string.replace('mystery', 'others')
        genre_string = genre_string.replace('action', 'others')
        return genre_string

    df['genres'] = df['genres'].apply(replace_genre)

    genre_dummies = df['genres'].str.get_dummies(sep=',')
    df = pd.concat([df, genre_dummies], axis=1)
    
    df['rated'] = df['rated'].replace('unrated', 'others')
    df['production_companies'] = df['production_companies'].replace('', 'no_company')

    # Let's add counts of languages, companies and countries as a new features
    df['num_spoken_languages'] = df['spoken_languages'].apply(lambda x: len(x.split(',')) if x else 0)
    df['num_production_companies'] = df['production_companies'].apply(lambda x: len(x.split(',')) if x else 0)
    df['num_production_countries'] = df['production_countries'].apply(lambda x: len(x.split(',')) if x else 0)
    df[['num_spoken_languages', 'num_production_companies', 'num_production_countries']].describe()

    df['is_foreign'] = df['production_countries'].apply(lambda x: 1 if 'us' not in x else 0)

    df['director_popularity'] = 0
    df['director_popularity_list'] = df.apply(lambda x: [], axis=1)
    df['director'] = df.apply(lambda x: [], axis=1)
    df['writer'] = df.apply(lambda x: [], axis=1)
    df['writer_popularity_list'] = df.apply(lambda x: [], axis=1)
    df['producer'] = df.apply(lambda x: [], axis=1)
    df['producer_popularity_list'] = df.apply(lambda x: [], axis=1)
    df['writer_popularity'] = 0
    df['producer_popularity'] = 0
    df['average_crew_popularity'] = 0
    df['number_crew_members'] = 0

    def update_popularity(row):
        director = []
        director_popularity_list = []
        writer = []
        writer_popularity_list = []
        producer = []
        producer_popularity_list = []
        director_popularity = 0
        writer_popularity = 0
        producer_popularity = 0
        total_popularity = 0
        num_directors = 0
        num_writers = 0
        num_producers = 0
        num_crew_members = len(row['crew'])
        
        for member in row['crew']:
            job = member.get('job', '')
            name = member.get('name', '')
            popularity = member.get('popularity', 0)
            total_popularity += popularity
            
            if job == 'director':
                director.append(name)
                director_popularity += popularity
                director_popularity_list.append(popularity)
                num_directors += 1
            elif job in ['writer', 'screenplay']:
                writer.append(name)
                writer_popularity += popularity
                writer_popularity_list.append(popularity)
                num_writers += 1
            elif job == 'producer':
                producer.append(name)
                producer_popularity += popularity
                producer_popularity_list.append(popularity)
                num_producers += 1
        
        # Average the popularity for directors, writers, and producers if there are multiple
        director_popularity /= max(num_directors, 1)
        writer_popularity /= max(num_writers, 1)
        producer_popularity /= max(num_producers, 1)
        
        # Compute the average popularity for all crew members
        average_crew_popularity = total_popularity / max(num_crew_members, 1)
        
        return pd.Series([director, director_popularity, director_popularity_list, writer, writer_popularity_list,writer_popularity,producer,  producer_popularity_list, producer_popularity, average_crew_popularity, num_crew_members])

    df[['director', 'director_popularity', 'director_popularity_list', 'writer', 'writer_popularity_list', 'writer_popularity', 'producer', 'producer_popularity_list','producer_popularity', 'average_crew_popularity', 'number_crew_members']] = df.apply(update_popularity, axis=1)

    df['average_cast_popularity'] = 0
    df['number_cast_members'] = 0

    def update_cast_popularity(row):
        total_popularity = 0
        top_cast_popularity = 0
        num_cast_members = len(row['cast'])
        
        # Sort the cast members by popularity, in descending order
        sorted_cast = sorted(row['cast'], key=lambda x: x.get('popularity', 0), reverse=True)
        
        for idx, member in enumerate(sorted_cast):
            popularity = member.get('popularity', 0)
            total_popularity += popularity
        
        # Compute the average popularity for all cast members
        average_cast_popularity = total_popularity / max(num_cast_members, 1)
        
        return pd.Series([average_cast_popularity, num_cast_members])

    df[['average_cast_popularity', 'number_cast_members']] = df.apply(update_cast_popularity, axis=1)

    return df[[ 'runtime',
      'collection',  
      'is_english',
      'log_adjusted_budget',
      'winter', 
      'animation', 
      'drama', 
      'moderate performing', 
      'others', 
      'num_spoken_languages',
      'num_production_companies', 
      'num_production_countries', 
      'is_foreign',
      'director_popularity', 
      'writer_popularity', 
      'producer_popularity', 
      'average_crew_popularity',
      'number_crew_members', 
      'average_cast_popularity', 
      'number_cast_members', 
      "numerical_ROI_category", 
      'numerical_rating_category', 
      'numerical_award_category'
      ]]



print("Read dataset data/movies.parquet")
df = pd.read_parquet('data/movies.parquet')
df = clean_data(df.copy())
dataset_df = features_extraction(df.copy())

#df.to_parquet('data/cleaned/selected_features.parquet')

labels = dataset_df[["numerical_ROI_category", 'numerical_rating_category', 'numerical_award_category']]
dataset_df.drop(["numerical_ROI_category", 'numerical_rating_category', 'numerical_award_category'], axis=1, inplace=True)

X_np = dataset_df.to_numpy()
y_np = labels.to_numpy()

X_full_train, y_full_train, X_test, y_test = iterative_train_test_split(X_np, y_np, test_size = 0.2)
X_train, y_train, X_val, y_val = iterative_train_test_split(X_full_train, y_full_train, test_size = 0.25)

X_train_df = pd.DataFrame(X_train, columns=dataset_df.columns)  
X_val_df = pd.DataFrame(X_val, columns=dataset_df.columns) 
X_test_df = pd.DataFrame(X_test, columns=dataset_df.columns) 

y_train_df = pd.DataFrame(y_train, columns=labels.columns)  
y_val_df = pd.DataFrame(y_val, columns=labels.columns) 
y_test_df = pd.DataFrame(y_test, columns=labels.columns) 

print("DictVectorizer")
dv = DictVectorizer(sparse=False)
X_train_df_t = dv.fit_transform(X_train_df.to_dict(orient='records'))
X_val_df_t = dv.transform(X_val_df.to_dict(orient='records'))
X_test_df_t = dv.transform(X_test_df.to_dict(orient='records'))

print("StandardScaler")
scaler = StandardScaler()
X_train_df_t = scaler.fit_transform(X_train_df_t)
X_val_df_t = scaler.transform(X_val_df_t)
X_test_df_t = scaler.transform(X_test_df_t)

train_pool = Pool(X_train_df_t, y_train)
val_pool = Pool(X_val_df_t, y_val)

print("Train CatBoostClassifier")
catboost_classifier = CatBoostClassifier(loss_function='MultiLogloss',
    eval_metric='HammingLoss',
    iterations=400, depth=6, learning_rate=0.1, random_state=42)
catboost_classifier.fit(train_pool, eval_set=val_pool, metric_period=10, plot=True, verbose=50)

print("Evaluate CatBoostClassifier")
val_predict = catboost_classifier.predict(X_val_df_t)
from catboost.utils import eval_metric
accuracy = eval_metric(y_val, val_predict, 'Accuracy')[0]
print(f'Accuracy: {accuracy}')

accuracy_per_class = eval_metric(y_val, val_predict, 'Accuracy:type=PerClass')
for cls, value in zip(catboost_classifier.classes_, accuracy_per_class):
    print(f'Accuracy for class {cls}: {value}')

hamming = eval_metric(y_val, val_predict, 'HammingLoss')[0]
print(f'HammingLoss: {hamming:.4f}')
mean_accuracy_per_class = sum(accuracy_per_class) / len(accuracy_per_class)
print(f'MeanAccuracyPerClass: {mean_accuracy_per_class:.4f}')
print(f'HammingLoss + MeanAccuracyPerClass = {hamming + mean_accuracy_per_class}')

for metric in ('Precision', 'Recall', 'F1'):
    print(metric)
    values = eval_metric(y_val, val_predict, metric)
    for cls, value in zip(catboost_classifier.classes_, values):
        print(f'class={cls}: {value:.4f}')
    print()

# Save the model
with open('models_binary/catboost_classifier_model.pkl', 'wb') as f_model:
    pickle.dump(catboost_classifier, f_model)

# Save the DictVectorizer
with open('models_binary/dict_vectorizer.pkl', 'wb') as f:
    pickle.dump(dv, f)

# Save the StandardScaler
with open('models_binary/standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)