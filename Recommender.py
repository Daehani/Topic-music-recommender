import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn.preprocessing as pp

def ratingMatrix(history_df):
    """
    Count user's listening event and normalilzed rating matrix
    Input
    ---------
    history_df : dataframe, listening event history dataframe
    Output
    -------
    user : list, user name list 
    item : list, item name list
    rating_norm_mat : sp_matrix, user-item l2 norm rating matrix
    """
    count_df = history_df.groupby(['u_id', 'track_id']).count().reset_index()[[0,1,2]]
    count_df.columns = ['u_id', 'track_id', 'count']
    rating_mat = count_df.pivot(index='u_id', columns='track_id', values='count')
    user_list = rating_mat.index
    item_list = rating_mat.columns
    
    # normalize rating matrix : L2 norm
    rating_mat = rating_mat.as_matrix()
    rating_mat[np.isnan(rating_mat)] = 0
    rating_norm_mat = pp.normalize(rating_mat, axis=1)
    
    return user_list, item_list, rating_norm_mat

def simCosMatrix(rating_mat, axis):
    """
    Compute cosine similarity  
    Input
    -----
    rating_mat : np_array, user-item rating matrix
    axis = int, 0(row, user similarity), 1(col, item similarity)
    Output
    -----
    sim_matrix : sp_matrix, user(or item) similarity matrix
    """
    rating_mat = sp.csr_matrix(rating_mat)

    if axis == 0:
        rating_mat = pp.normalize(rating_mat, axis=1)
        rating_mat = rating_mat.astype('float16')
        sim_matrix = rating_mat * rating_mat.T
        return sim_matrix
    
    elif axis == 1:
        rating_mat = rating_mat.tocsc()
        rating_mat = pp.normalize(rating_mat, axis=0)
        rating_mat = rating_mat.astype('float16')
        sim_matrix = rating_mat.T * rating_mat
        return sim_matrix

def simNbrMatrix(sim_matrix, k):
    """
    Store only similarity of user(or item) neighbors   
    Input
    -----
    sim_matrix : sp_matrix, user(or item) similarity matrix
    k : int, the number of neighbor 
    Output
    -----
    nbr_sim_matrix : sp_matrix, user(or item)'s neighbor similarity matrix 
    """
    nbr_sim_matrix = sp.lil_matrix(sim_matrix.shape)
    
    for i in range(sim_matrix.shape[0]):

        sim_matrix[i, i] = 0 # Self similarity = 0
        neighbor_Ind = np.argsort(sim_matrix[i, :].toarray())[0][::-1][:k]
        nbr_sim_matrix[i, neighbor_Ind] = sim_matrix[i, neighbor_Ind]
    
    return nbr_sim_matrix.tocsr()
    
def topnList(score_matrix, user_list, item_list, n):
    """
    Recommend top n items
    Input
    -----
    score_matrix : sp_matrix, user-item score matrix
    user_list : list, user name list 
    item_list : list, item name list 
    n : int, the number of recommedation item 
    Output
    -----
    result_df : pd_dataframe, recommended n items by user
    """
    result_df = pd.DataFrame()
    
    for i in range(score_matrix.shape[0]):

        user_score = score_matrix[i, :]
        item_ind = user_score.indices
        top_item_ind = np.argsort(user_score.toarray()[0])[::-1][:n]
        top_item_ind = [ind for ind in top_item_ind if ind in item_ind]
        if top_item_ind != []:
            top_n_item = [item_list[i] for i in top_item_ind]
            user_result_dic = {'u_id' : user_list[i], 'track_id' : top_n_item, 'rank' : [int(i+1) for i in range(len(top_n_item))], 'score' : user_score[0, top_item_ind].toarray()[0]}
            top_n_item_df = pd.DataFrame(user_result_dic)
            result_df = result_df.append(top_n_item_df)

    return result_df[[3,2,1,0]]
    
def recommenderCF(rating_matrix, sim_matrix, user_list, item_list, cf_type, n, k):
    """
    User-based Collaborative Filtering, Item-based Collaborative Filtering
    Input
    -----
    rating_matrix : sp_matrix, user-item rating matrix
    sim_matrix : sp_matrix, user(or item)'s similarity matrix 
    user_list : list, user name list 
    item_list : list, item name list 
    cf_type : str, user or item
    k : int, the number of neighbor 
    n : int, the number of recommedation item 
    Output
    ------
    score_matrix : sp_matrix, user-item score matrix
    result_df : pd_dataframe, recommended n items by user
    """
    score_matrix = sp.lil_matrix(rating_matrix.shape)

    
    # User-based CF
    if cf_type == 'user':
         
        rating_matrix = sp.csr_matrix(rating_matrix)

        # select neighborhood users
        print("Select neighborhood users...")
        nbr_sim_matrix = simNbrMatrix(sim_matrix, k)
              
        # compute predictions
        print("Compute predictions...")       
        for user_i in range(rating_matrix.shape[0]):

            user_similarity = nbr_sim_matrix[user_i, :]
            rec_item_ind = [i for i in range(rating_matrix.shape[1]) if i not in rating_matrix[user_i, :].indices.tolist()]
            
            for item_i in rec_item_ind:
                
                item_rating = rating_matrix[:, item_i]

                # Score prediction
                user_similarity_ind = user_similarity.nonzero()[1]
                item_rating_ind = item_rating.nonzero()[0]
                share_item = list(set(user_similarity_ind).intersection(item_rating_ind))
                if  share_item != []:
                    score = user_similarity.dot(item_rating).data[0] / user_similarity[:, share_item].sum()
                    score_matrix[user_i, item_i] = score


    # Item-based CF
    elif cf_type == 'item':
        
        rating_matrix = sp.csc_matrix(rating_matrix)

        # select neighborhood items
        print("Select neighborhood items...")     
        nbr_sim_matrix = simNbrMatrix(sim_matrix, k)
              
        # compute predictions
        print("Compute predictions...")
        for item_i in range(rating_matrix.shape[1]):

            item_similarity = nbr_sim_matrix[item_i, :]
            rec_user_ind = [i for i in range(rating_matrix.shape[0]) if i not in rating_matrix[:, item_i].indices.tolist()]

            for user_i in rec_user_ind:
                user_rating = rating_matrix[user_i, :]

                # Score prediction
                item_similarity_ind = item_similarity.nonzero()[1]
                user_rating_ind = user_rating.nonzero()[1]
                share_item = list(set(item_similarity_ind).intersection(user_rating_ind))
                if  share_item != []:
                    score = user_rating.dot(item_similarity.T).data[0] / item_similarity[:, share_item].sum()
                    score_matrix[user_i, item_i] = score
    
    score_matrix = score_matrix.tocsr()
        
    # Recommend top-n item
    print("Recommend Top-n item...")  
    result_df = topnList(score_matrix, user_list, item_list, n)
    
    return score_matrix , result_df

def hybridScore(score_matrix_1, score_matrix_2, c):
    """
    Calculate weighted average score matrix 1 and score matrix 2
    Input
    -----
    score_matrix_1 : sp_matrix, user-item score matrix
    score_matrix_2 : sp_matrix, user-item score matrix
    c : float, hybrid parameter
    Output
    -----
    hybrid_score_matrix : sp_matrix, user-item hybrid score matrix
    """
    norm_score_matrix_1 = pp.MaxAbsScaler().fit_transform(score_matrix_1.T).T
    norm_score_matrix_2 = pp.MaxAbsScaler().fit_transform(score_matrix_2.T).T
    return (1-c) * norm_score_matrix_1 + c * norm_score_matrix_2