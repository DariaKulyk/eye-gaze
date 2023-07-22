""" This code analyzes SemArt dataset which contains painting desciptions 
to determine most common words for different artistic genres """

import pandas as pd
import numpy as np
import nltk
import scipy.cluster.hierarchy as shc 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

nlp = spacy.load("en_core_web_lg")


nltk.download('punkt')  
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('maxent_ne_chunker')

# Load the dataset
df = pd.read_csv('../../data/external/semart_train.csv', sep='\t', encoding = 'latin-1')
# print(df.head(15))

filtered_df = df[(df['TIMEFRAME'] == '1601-1650') | (df['TIMEFRAME'] == '1651-1700')] 

# Get unique genres of paintings and create plural versions of genre names
unique_genres = filtered_df["TYPE"].unique()
plural_genres = np.char.add(unique_genres.astype(str), 's')

grouped_by_type = filtered_df.groupby('TYPE')

# Define words to remove
stop_words = set(stopwords.words('english'))
# Manually defined words to remove
individual_words = (["painting", "picture", "work", "artist", "painter", "scene", "figure", "time", "year", "century", "wall", "subject", "style", "art", "van", "composition", "theme", "version", "example", "background", "life", "number", "colour", "foreground", "right", "left", "rembrandt", "period", "age", "part", "group", "event", "history", "view", "cycle", "space", "light", "series", "story", "allegory", "way", "side", "sketch", "design", "models", "de", "studio", "panel", "collection", "effect", "depiction", "type", "motif", "viewer", "piece", "end", "da", "perspective", "der", "distance", "influence", "form", "tradition", "setting", "zquez", "set", "de'", "place", "drawing", "canvas", "object", "sitter", "sale", "member", "self-portrait", "length"]) 
remove_words = stop_words.union(individual_words, plural_genres, unique_genres)

is_noun = lambda pos: pos[:2] == 'NN'
tokenizer = RegexpTokenizer(r"\w+['\w-]*")
lemmatizer = nltk.stem.WordNetLemmatizer()

def is_name(word):
    return all([w not in word for w in word.lower().split()])


for _type, group in grouped_by_type:
    descriptions = ' '.join(group['DESCRIPTION'])
    tokens = tokenizer.tokenize(descriptions)
    nouns = [word for (word, pos) in nltk.pos_tag(tokens) if is_noun(pos) and not is_name(word)]
    lemmas = [lemmatizer.lemmatize(t) for t in nouns]
    filtered_tokens = [word for word in lemmas if word.lower() not in remove_words]
    fdist = FreqDist(filtered_tokens)
    most_common = fdist.most_common(15) 
    print(f"Type {_type}:")
    for term, frequency in most_common:
        print(f"{term}: {frequency} occurrences")
    print("----------------------------------------")
    
    # Grouping object categories
    object_categories = [term for term, _ in most_common]
    embeddings = [nlp(category).vector for category in object_categories]
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, affinity='cosine', linkage='average')
    clusters = clustering.fit_predict(embeddings)
    
    grouped_categories = {}
    for category, cluster in zip(object_categories, clusters):
        if cluster not in grouped_categories:
            grouped_categories[cluster] = []
        grouped_categories[cluster].append(category)
        
    print(f"Type {_type}:")
    for group, categories in grouped_categories.items():
        print(f"Group {group}: {categories}")
    print("----------------------------------------")
    
    