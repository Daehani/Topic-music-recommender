# Topic-music-recommender
Music Recommender System using Topic Modeling


```python
import numpy as np
import pandas as pd
import lda

from TopicModeling import *
from Recommender import *
```

# 1. Data


```python
listen_df = pd.read_csv('Lastfm_listening_data.csv', encoding = "ISO-8859-1")
```

## Tag


```python
track_df = itemDataframe(listen_df)
lastfm_network = lastfmAuth()
tracktag_dic = lastfmCrawlTag(track_df, lastfm_network)
track, tag, tracktag_m = tracktagMatrix(tracktag_dic)
```


```python
listen_df = listen_df[listen_df['track_id'].isin(track)]
```


```python
listen_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>u_id</th>
      <th>time</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>track_id</th>
      <th>track_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>user_000031</td>
      <td>2009-05-01 10:37:23</td>
      <td>9c1ff574-2ae4-4fea-881f-83293d0d5881</td>
      <td>...And You Will Know Us By The Trail Of Dead</td>
      <td>fe8be4ac-bbe8-48f3-bcac-25f45cb7d75e</td>
      <td>Source Tags And Codes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>user_000291</td>
      <td>2009-05-01 09:02:09</td>
      <td>6aa40207-fec8-43a7-991d-b872a42def05</td>
      <td>Amy Macdonald</td>
      <td>21bd2851-63a2-455b-800b-ea1aa4900a99</td>
      <td>This Is The Life</td>
    </tr>
    <tr>
      <th>2</th>
      <td>user_000291</td>
      <td>2009-05-01 09:05:37</td>
      <td>83d91898-7763-47d7-b03b-b92132375c47</td>
      <td>Pink Floyd</td>
      <td>feecff58-8ee2-4a7f-ac23-dc8ce7925286</td>
      <td>Wish You Were Here</td>
    </tr>
    <tr>
      <th>3</th>
      <td>user_000291</td>
      <td>2009-05-01 09:12:12</td>
      <td>83d91898-7763-47d7-b03b-b92132375c47</td>
      <td>Pink Floyd</td>
      <td>2d982a74-a21a-4714-9d81-405ede915053</td>
      <td>Comfortably Numb</td>
    </tr>
    <tr>
      <th>4</th>
      <td>user_000291</td>
      <td>2009-05-01 09:22:06</td>
      <td>a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432</td>
      <td>U2</td>
      <td>44135d7f-f5d6-4ef9-8518-ed11f5b705e7</td>
      <td>One</td>
    </tr>
  </tbody>
</table>
</div>




```python
listen_df.u_id.nunique()
```




    10




```python
listen_df.track_id.nunique()
```




    182



# 2. Topic Modeilng - LDA


```python
LDAModel = lda.LDA(n_topics=30)
LDAModel.fit(tracktag_m.astype(int))
```




    <lda.lda.LDA at 0x2af8a7ebc18>




```python
tracktopic_m = LDAModel.doc_topic_
topictag_m = LDAModel.topic_word_
```


```python
n_top_words = 10
for i, topic_dist in enumerate(topictag_m):
    topic_words = np.array(tag)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
    print('Topic {}, {}'.format(i+1, ','.join(topic_words)))
```

    Topic 1, metal,alternative metal,nu metal,soundtrack,rock,deftones,alternative,hard rock,system of a down,queen of the damned
    Topic 2, beautiful,love,mellow,sad,melancholic,indie,melancholy,romantic,shoegaze,ballad
    Topic 3, alternative,rock,emo,evanescence,sad,melancholy,one tree hill,alternative rock,epic,emotional
    Topic 4, folk,indie,singer-songwriter,indie folk,acoustic,happy,comedy,gypsy,alternative,anti-folk
    Topic 5, electronic,electronica,dance,new rave,electro,indie,british,electropop,indietronica,electroclash
    Topic 6, gothic,industrial,gothic rock,electronic,female vocalists,synth-rock,darkwave,synthpop,gothic metal,synth rock
    Topic 7, hip-hop,electronic,dance,electronica,electro,crunk,alternative,pop,electro rap,richman
    Topic 8, rock,alternative,alternative rock,emo,one tree hill,jimmy eat world,indie,love,awesome,muse
    Topic 9, placebo,alternative,british,indie,britpop,rock,heard on pandora,indie rock,modern rock,all time favourites
    Topic 10, heavy metal,hard rock,metal,rock,speed metal,motorhead,classic rock,power metal,nwobhm,rock n roll
    Topic 11, english,60s,catchy,60's,pop,oldies,greatest songs ever,rock,beach,1960s
    Topic 12, classic rock,rock,60s,the beatles,british,pop,beatles,oldies,psychedelic,classic
    Topic 13, progressive rock,psychedelic,psychedelic rock,rock,pink floyd,progressive,british,70s,led zeppelin,art rock
    Topic 14, industrial,industrial rock,industrial metal,electronic,nin,nine inch nails,trent reznor,rock,cover,hard rock
    Topic 15, rock,indie rock,indie,alternative,alternative rock,u2,the killers,favorites,00s,pop
    Topic 16, chillout,trip-hop,electronic,the best,downtempo,female vocalists,canadian,chill,broken social scene,electronica
    Topic 17, pop,80s,synthpop,electronic,new wave,country,dance,taylor swift,depeche mode,synth pop
    Topic 18, soundtrack,halloween,film music,tango,crunk rock,balkan,christian rock,tarantino,party music,crunk
    Topic 19, acoustic,instrumental,post-rock,indian burial ground music,icelandic,milow,roots,aussie,heard on pandora,guitar
    Topic 20, doom metal,traditional doom metal,march,classic rock,embrace chaos,rocking,comedy,feelgood tracks,thirteenth step,heavy fucking metal
    Topic 21, hip hop,hip-hop,german,hiphop,rap,german hiphop,deutsch,deutscher hip hop,hallo,stoned
    Topic 22, rock,alternative rock,alternative,90s,seen live,grunge,favorites,00s,awesome,favourite songs
    Topic 23, 60s,pop,oldies,classic rock,rock,surf,beach boys,surf rock,summer,the beach boys
    Topic 24, ambient,instrumental,post-rock,awesome,summer,electronic rock,electro-rock-pop,jams,head automatica,experimental
    Topic 25, indie,alternative,rock,indie rock,favorite songs,indie-rock,art rock,songs that deserve to be tagged yet can not be sufficiently described in a short manner,pop,favourite songs
    Topic 26, pop,dance,britney spears,female vocalists,rnb,rihanna,catchy,00s,sexy,lady gaga
    Topic 27, female vocalists,indie,pop,singer-songwriter,beautiful,female vocalist,alternative,mellow,piano,female
    Topic 28, 60s,summer,oldies,california,pop,american,surf,beach boys,best song ever,male vocalists
    Topic 29, industrial,instrumental,electronic,classic british metal,classic british hard rock,classic british rock,british rock,british hard rock,classic hard rock,new wave of british heavy metal
    Topic 30, rock,alternative,cover,alternative rock,punk,polish punk,blues rock,hard rock,southern rock,punk rock
    


```python
tracktopic_m.shape
```




    (182, 30)




```python
topictag_m.shape
```




    (30, 4092)



# 3. Recommendation

## Item-based Collaborative Filtering


```python
user, item, rating_m = ratingMatrix(listen_df)
item_sim_m = simCosMatrix(rating_m, axis=1)
```


```python
score_cf_m, result_cf_df = recommenderCF(rating_m, item_sim_m, user, item, cf_type='item', n=10, k=182)
```

    Select neighborhood items...
    Compute predictions...
    Recommend Top-n item...
    

## Topic-based Content-based Filtering


```python
topic_sim_m = simCosMatrix(tracktopic_m, axis=0)
score_cbf_m, result_cbf_df = recommenderCF(rating_m, topic_sim_m, user, item, cf_type='item', n=10, k=182)
```

    Select neighborhood items...
    Compute predictions...
    Recommend Top-n item...
    

## Hybird Recommendation


```python
score_hybrid_m = hybridScore(score_cf_m, score_cbf_m, c=0.1)
result_hybrid_df = topnList(score_hybrid_m, user, item, 10)
```

# 4. Top-n Recommendation List

## Item-based Collaborative Filtering


```python
result_cf_df = result_cf_df.merge(track_df, on='track_id').sort_values(['u_id', 'rank'])
```


```python
result_cf_df[result_cf_df['u_id'] == "user_000524"]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>u_id</th>
      <th>track_id</th>
      <th>score</th>
      <th>rank</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>track_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>user_000524</td>
      <td>7ea0d658-b2a9-4a8d-a346-48fb2b8acf6a</td>
      <td>0.099504</td>
      <td>1</td>
      <td>b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d</td>
      <td>The Beatles</td>
      <td>Here Comes The Sun</td>
    </tr>
    <tr>
      <th>15</th>
      <td>user_000524</td>
      <td>09a0040e-fd69-4dc9-aa1f-22311108a964</td>
      <td>0.099504</td>
      <td>2</td>
      <td>3ac2a4a2-52b3-498b-bbc8-31443c68dfe0</td>
      <td>Missy Higgins</td>
      <td>The River</td>
    </tr>
    <tr>
      <th>16</th>
      <td>user_000524</td>
      <td>210005fc-3b04-4112-9b39-5741244f71c7</td>
      <td>0.099504</td>
      <td>3</td>
      <td>b10bbbfc-cf9e-42e0-be17-e2c3e1d2600d</td>
      <td>The Beatles</td>
      <td>Maxwell'S Silver Hammer</td>
    </tr>
    <tr>
      <th>17</th>
      <td>user_000524</td>
      <td>21bd2851-63a2-455b-800b-ea1aa4900a99</td>
      <td>0.099504</td>
      <td>4</td>
      <td>6aa40207-fec8-43a7-991d-b872a42def05</td>
      <td>Amy Macdonald</td>
      <td>This Is The Life</td>
    </tr>
    <tr>
      <th>18</th>
      <td>user_000524</td>
      <td>97a521ad-9036-408b-9572-e1d63b872e06</td>
      <td>0.099504</td>
      <td>5</td>
      <td>7952b266-9fd4-4a09-a324-7dc84f11b5fc</td>
      <td>The John Butler Trio</td>
      <td>What You Want</td>
    </tr>
    <tr>
      <th>19</th>
      <td>user_000524</td>
      <td>94a2565f-9dd3-46bd-a70f-80c41224f561</td>
      <td>0.099504</td>
      <td>6</td>
      <td>a3cb23fc-acd3-4ce0-8f36-1e5aa6a18432</td>
      <td>U2</td>
      <td>Stay (Faraway, So Close!)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>user_000524</td>
      <td>171ad8ee-51ae-48c3-ae30-35436b2411af</td>
      <td>0.099504</td>
      <td>7</td>
      <td>a4e34a43-d8de-48eb-9ec6-349b62756590</td>
      <td>T?l?phone</td>
      <td>Un Autre Monde</td>
    </tr>
    <tr>
      <th>21</th>
      <td>user_000524</td>
      <td>8d69d3fc-5e0e-4c84-97eb-1891ca0d66e8</td>
      <td>0.099504</td>
      <td>8</td>
      <td>a41ac10f-0a56-4672-9161-b83f9b223559</td>
      <td>Van Morrison</td>
      <td>Brown Eyed Girl</td>
    </tr>
    <tr>
      <th>22</th>
      <td>user_000524</td>
      <td>8b6d9029-1990-4660-bd57-24b5bc0fb626</td>
      <td>0.099504</td>
      <td>9</td>
      <td>d4d17620-fd97-4574-92a8-a2cb7e72ce42</td>
      <td>The Verve</td>
      <td>Bitter Sweet Symphony</td>
    </tr>
    <tr>
      <th>23</th>
      <td>user_000524</td>
      <td>ba7ed38e-f5b0-438c-9df3-d5e061c353dd</td>
      <td>0.099504</td>
      <td>10</td>
      <td>7952b266-9fd4-4a09-a324-7dc84f11b5fc</td>
      <td>The John Butler Trio</td>
      <td>Ocean</td>
    </tr>
  </tbody>
</table>
</div>



## Topic-based Content-based Filtering


```python
result_cbf_df = result_cbf_df.merge(track_df, on='track_id').sort_values(['u_id', 'rank'])
```


```python
result_cbf_df[result_cbf_df['u_id'] == "user_000524"]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>u_id</th>
      <th>track_id</th>
      <th>score</th>
      <th>rank</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>track_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>user_000524</td>
      <td>5b1e09e5-5f6f-447f-815b-57607acf6339</td>
      <td>0.225121</td>
      <td>1</td>
      <td>8c9200b8-8e05-41d5-836e-44a37905560e</td>
      <td>Hadouken!</td>
      <td>Get Smashed Gate Crash</td>
    </tr>
    <tr>
      <th>76</th>
      <td>user_000524</td>
      <td>09a0040e-fd69-4dc9-aa1f-22311108a964</td>
      <td>0.220694</td>
      <td>2</td>
      <td>3ac2a4a2-52b3-498b-bbc8-31443c68dfe0</td>
      <td>Missy Higgins</td>
      <td>The River</td>
    </tr>
    <tr>
      <th>78</th>
      <td>user_000524</td>
      <td>08ddb3fb-8f4c-4ea2-adaf-06d7df53e155</td>
      <td>0.208642</td>
      <td>3</td>
      <td>e795e03d-b5d5-4a5f-834d-162cfb308a2c</td>
      <td>Pj Harvey</td>
      <td>This Is Love</td>
    </tr>
    <tr>
      <th>15</th>
      <td>user_000524</td>
      <td>5447420a-88fd-4f5c-a293-bda90cbc0f44</td>
      <td>0.204960</td>
      <td>4</td>
      <td>89618a45-ff4a-4e5f-942e-3ef93c8c555c</td>
      <td>Witchfinder General</td>
      <td>Satan'S Children (Live)</td>
    </tr>
    <tr>
      <th>79</th>
      <td>user_000524</td>
      <td>06d6f9ad-401b-4215-8304-e7af3f3692b4</td>
      <td>0.191193</td>
      <td>5</td>
      <td>b7ffd2af-418f-4be2-bdd1-22f8b48613da</td>
      <td>Nine Inch Nails</td>
      <td>Wish</td>
    </tr>
    <tr>
      <th>80</th>
      <td>user_000524</td>
      <td>910a40bb-3b77-48cb-b978-77eae3d0398f</td>
      <td>0.191093</td>
      <td>6</td>
      <td>efef848b-63e4-4323-8ef7-69a48fbdd51d</td>
      <td>4 Non Blondes</td>
      <td>What'S Up</td>
    </tr>
    <tr>
      <th>81</th>
      <td>user_000524</td>
      <td>9839b527-a36d-4efb-8682-526601be5131</td>
      <td>0.165318</td>
      <td>7</td>
      <td>8ca01f46-53ac-4af2-8516-55a909c0905e</td>
      <td>My Bloody Valentine</td>
      <td>Sometimes</td>
    </tr>
    <tr>
      <th>82</th>
      <td>user_000524</td>
      <td>9391e18a-a0f4-443c-9343-69af2849abd5</td>
      <td>0.153321</td>
      <td>8</td>
      <td>95e1ead9-4d31-4808-a7ac-32c3614c116b</td>
      <td>The Killers</td>
      <td>Daddy'S Eyes</td>
    </tr>
    <tr>
      <th>62</th>
      <td>user_000524</td>
      <td>924de9ff-b23a-40ef-ae20-153d42472e6d</td>
      <td>0.152915</td>
      <td>9</td>
      <td>847e8284-8582-4b0e-9c26-b042a4f49e57</td>
      <td>Placebo</td>
      <td>Hare Krishna</td>
    </tr>
    <tr>
      <th>83</th>
      <td>user_000524</td>
      <td>910e91d3-e711-4b49-9de4-0caf5c536174</td>
      <td>0.152220</td>
      <td>10</td>
      <td>89618a45-ff4a-4e5f-942e-3ef93c8c555c</td>
      <td>Witchfinder General</td>
      <td>Soviet Invasion</td>
    </tr>
  </tbody>
</table>
</div>



## Hybird Recommendation


```python
result_hybrid_df = result_hybrid_df.merge(track_df, on='track_id').sort_values(['u_id', 'rank'])
```


```python
result_hybrid_df[result_hybrid_df['u_id'] == "user_000524"]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>u_id</th>
      <th>track_id</th>
      <th>score</th>
      <th>rank</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>track_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>user_000524</td>
      <td>5b1e09e5-5f6f-447f-815b-57607acf6339</td>
      <td>1.000000</td>
      <td>1</td>
      <td>8c9200b8-8e05-41d5-836e-44a37905560e</td>
      <td>Hadouken!</td>
      <td>Get Smashed Gate Crash</td>
    </tr>
    <tr>
      <th>76</th>
      <td>user_000524</td>
      <td>09a0040e-fd69-4dc9-aa1f-22311108a964</td>
      <td>0.998034</td>
      <td>2</td>
      <td>3ac2a4a2-52b3-498b-bbc8-31443c68dfe0</td>
      <td>Missy Higgins</td>
      <td>The River</td>
    </tr>
    <tr>
      <th>78</th>
      <td>user_000524</td>
      <td>08ddb3fb-8f4c-4ea2-adaf-06d7df53e155</td>
      <td>0.992680</td>
      <td>3</td>
      <td>e795e03d-b5d5-4a5f-834d-162cfb308a2c</td>
      <td>Pj Harvey</td>
      <td>This Is Love</td>
    </tr>
    <tr>
      <th>79</th>
      <td>user_000524</td>
      <td>06d6f9ad-401b-4215-8304-e7af3f3692b4</td>
      <td>0.984929</td>
      <td>4</td>
      <td>b7ffd2af-418f-4be2-bdd1-22f8b48613da</td>
      <td>Nine Inch Nails</td>
      <td>Wish</td>
    </tr>
    <tr>
      <th>80</th>
      <td>user_000524</td>
      <td>910a40bb-3b77-48cb-b978-77eae3d0398f</td>
      <td>0.984885</td>
      <td>5</td>
      <td>efef848b-63e4-4323-8ef7-69a48fbdd51d</td>
      <td>4 Non Blondes</td>
      <td>What'S Up</td>
    </tr>
    <tr>
      <th>81</th>
      <td>user_000524</td>
      <td>9839b527-a36d-4efb-8682-526601be5131</td>
      <td>0.973435</td>
      <td>6</td>
      <td>8ca01f46-53ac-4af2-8516-55a909c0905e</td>
      <td>My Bloody Valentine</td>
      <td>Sometimes</td>
    </tr>
    <tr>
      <th>82</th>
      <td>user_000524</td>
      <td>9391e18a-a0f4-443c-9343-69af2849abd5</td>
      <td>0.968106</td>
      <td>7</td>
      <td>95e1ead9-4d31-4808-a7ac-32c3614c116b</td>
      <td>The Killers</td>
      <td>Daddy'S Eyes</td>
    </tr>
    <tr>
      <th>83</th>
      <td>user_000524</td>
      <td>924de9ff-b23a-40ef-ae20-153d42472e6d</td>
      <td>0.967926</td>
      <td>8</td>
      <td>847e8284-8582-4b0e-9c26-b042a4f49e57</td>
      <td>Placebo</td>
      <td>Hare Krishna</td>
    </tr>
    <tr>
      <th>84</th>
      <td>user_000524</td>
      <td>290dcfd6-827e-4864-b71a-553c45ba526b</td>
      <td>0.966527</td>
      <td>9</td>
      <td>99acd557-c4e2-4086-9be9-85f57184dadc</td>
      <td>O.S.T.R.</td>
      <td>Pocz?tek...</td>
    </tr>
    <tr>
      <th>85</th>
      <td>user_000524</td>
      <td>1e21e59e-ae7c-4658-bfe4-1e0c749d19c4</td>
      <td>0.965959</td>
      <td>10</td>
      <td>c3f28da8-662d-4f09-bdc7-3084bf685930</td>
      <td>Iron &amp; Wine</td>
      <td>Free Until They Cut Me Down</td>
    </tr>
  </tbody>
</table>
</div>


