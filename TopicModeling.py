import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pylast as pl
import re
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
import collections
import scipy.sparse as sp

def itemDataframe(history_df):
    """
    Create unique item's information dataframe from user's listening dataframe
    Input
    -----
    history_df : pd_dataframe, user's listening event
    Output
    -----
    item_df : pd_dataframe, all unique items
    """
    item = pd.DataFrame(history_df['track_id'].sort_values().unique(), columns=['track_id'])
    item = item[~pd.isnull(item['track_id'])]
    item_df = history_df[~pd.isnull(history_df['track_id'])][[2,3,4,5]].drop_duplicates(['track_id']).sort_values(by=['artist_name','track_id'])
    return item_df

def lastfmAuth():
    """
    Last.fm API Authorization using account
    Input
    -----
    Output
    -----
    network : last.fm API object
    """
    # account info.
    API_KEY = "49f943150c648e3d408c4c3948ea42a9"
    API_SECRET = "d0c424dd95539d644b2894f9eb2199bb"
    username = "Daehani"
    password_hash = pl.md5("a123!@#")
    
    # create last.fm obeject
    network = pl.LastFMNetwork(api_key = API_KEY, api_secret = API_SECRET, username = username, password_hash = password_hash)
    return network

def lastfmCrawlTag(item_df, lastfm_network):
    """
    Crawling tags of tracks in Last.fm
    Input
    -----
    item_df : pd_dataframe, all unique items
    lastfm_network : last.fm API object
    Object
    -----
    tracktag_dic_list : list, tag's track dictionary 
    """
    tracktag_dic_list = []
    
    for i in range(item_df.shape[0]):
        track_id = item_df.iloc[i]['track_id']
        track = item_df.iloc[i]['track_name']
        artist = item_df.iloc[i]['artist_name']
        tag_dic = {}

        # extract tag
        lastfm_track = lastfm_network.get_track(artist=artist, title=track)
        try:
            lastfm_tags = lastfm_track.get_top_tags()
        except:
            continue
        
        # create dictionary
        for tag in lastfm_tags:
            tag_dic.update({tag.item.get_name().lower():tag.weight}) 
        tag_dic.update({'track_id' : track_id})
        tracktag_dic_list.append(tag_dic)

        i += 1
    return tracktag_dic_list

def tracktagMatrix(tracktag_dic_list):
    """
    Create track-tag matrix
    Input
    -----
    tracktag_dic_list : list, tag's track dictionary 
    Output
    -----
    track_list : list, tracks from which tag was extracted
    tag_list : list, extracted tags
    tracktag_matrix : np.array, track-tag matrix
    """
    track_list = [dic['track_id'] for dic in tracktag_dic_list]
    
    # delete track name & artist name
    for dic in tracktag_dic_list:
        if dic['track_id']: 
            del dic['track_id']
            
        # score type: str -> int
        for key in dic.keys():
            dic[key] = int(dic[key])
            
    v = DictVectorizer(sparse=True)
    tracktag_matrix = v.fit_transform(tracktag_dic_list)
    tag_list = v.get_feature_names()
    return (track_list, tag_list, tracktag_matrix)

def tagNormalize(tag, stemmer='porter'):
    """
    Noramlize tag
    Input
    -----
    tag : str
    stemmer : porter(PorterStemmer), snowball(SnowballStemmer), wordnet(WordNetLemmatizer)
    Output
    -----
    stem_tag : str, stemmed tag
    """
    # &, n
    letters = re.sub('\&', ' and ', tag)
    letters = re.sub('\'n\'', ' and ', letters)
    letters = re.sub('\'n', ' and ', letters)
    letters = re.sub('n\'', ' and ', letters)
    letters = re.sub('n\'', ' and ', letters)
    letters = re.sub('-n-', ' and ', letters)
    letters = re.sub(' n ', ' and ', letters)
    
    # punctuation
    letters = re.sub('\'', '', letters)
    letters = re.sub('[^0-9a-zA-Z+]', " ", letters)

    words = letters.split()
    
    # decade
    words = [re.sub('^20s$', "1920s", word) for word in words]
    words = [re.sub('^30s$', "1930s", word) for word in words]
    words = [re.sub('^40s$', "1940s", word) for word in words]
    words = [re.sub('^50s$', "1950s", word) for word in words]
    words = [re.sub('^60s$', "1960s", word) for word in words]
    words = [re.sub('^70s$', "1970s", word) for word in words]
    words = [re.sub('^80s$', "1980s", word) for word in words]
    words = [re.sub('^90s$', "1990s", word) for word in words]
    words = [re.sub('^00s$', "2000s", word) for word in words]
    words = [re.sub('^10s$', "2010s", word) for word in words]
                   
    # Stemming
    if stemmer == 'porter':
        stem_words = [PorterStemmer().stem(word) for word in words]
    elif stemmer == 'snowball':
        stem_words = [SnowballStemmer('english').stem(word) for word in words]
    elif stemmer == 'wordnet':
        stem_words = [WordNetLemmatizer().lemmatize(word) for word in words]
    stem_tag = " ".join(stem_words)
    
    return stem_tag

def tracktagNormalize(tracktag_matrix, tag_list):
    """
    Noramlize track-tag matrix
    Input
    -----
    tracktag_matrix : sp_matrix, track-tag matrix
    tag_list : list
    Output
    -----
    tracktag_norm_matrix : sp_matrix, normalized track-tag matrix
    tag_norm_list : list,  normalized tag list
    """
    # normalizing Tag
    tag_norm_list = [tagNormalize(t, 'porter') for t in tag_list]
    
    # duplicated tag
    duplicate_tag_dic = collections.defaultdict(list)
    for i, t in enumerate(tag_norm_list):
        duplicate_tag_dic[t].append(i)
        
    duplicate_tag_dic = {t:ind_list for t, ind_list in duplicate_tag_dic.items() if len(ind_list) > 1}
    tag_dup = sorted(duplicate_tag_dic.keys())
    
    # remove duplicated tag columns in tracktag matrix
    tracktag_tmp = tracktag_matrix.tolil()
    tag_rm_ind_list = []
    for t in tag_dup:
        ind_list = duplicate_tag_dic[t]

        if t == "":
            tag_rm_ind_list.extend(ind_list)
            continue

        for i, tag_ind in enumerate(ind_list):
            if i == 0:
                tag_1st_ind = tag_ind
                continue
            else:    
                tracktag_tmp[:, tag_1st_ind] += tracktag_tmp[:, tag_ind]
                tag_rm_ind_list.append(tag_ind)
    
    tag_ind_list = [ind for ind in range(tracktag_tmp.shape[1]) if ind not in tag_rm_ind_list]
    tag_norm_list = [tag_ind_list[i] for i in tag_ind_list]
    tracktag_norm_matrix = tracktag_matrix[:, tag_ind_list]
   
    return tag_norm_list, tracktag_norm_matrix