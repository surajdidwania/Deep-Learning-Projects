# -*- coding: utf-8 -*-

import pandas as pd
data = pd.read_csv('http://bit.ly/uforeports', usecols = ['City','State'],nrows = 3,dtype = {'City':float})
data['Location'] = data.City + ',' + data.State
data.describe(include = ['object'])
data.shape
data.dtypes
data.columns
data.rename(columns = {'Colors Reported':'Colors_Reported','Shape Reported':'Shape_Reported'})
data.columns = data.columns.str.replace(' ','_')
data.columns
data.drop('Colors_Reported',axis = 1) #axis = 0 if rows delete
data['City'].sort_values()
data.sort_values(['City','Colors_Reported'])
is_long = movies.ratings >=200
is_long
data[is_long]
data[data.ratings >=3]['genre']
data.loc[data.ratings >=3, 'genre']
data[(data.ratings >=200) & (data.genre == 'Drama')]
data[data.genre.isin(['Crime','Drame'])]
for index,rows in data.iterrows():
    print(rows.City, rows.State)
data.select_dtypes(include = [np.number]).dtypes

#How do I use string methods in pandas?
'hello'.upper()

data.City.str.upper() # str method in dataframe
data.City.str.contains('')
data.City.str.replace('[','').str.replace(']','')


#How do I change the data type of a pandas Series?
data['City'] = data.City.astype(float) #to do math in column

#When should I use a "groupby" in pandas?
data.groupby('City').State.mean() .min() .max() .agg(['count','min','max','mean'])
data.groupby('City').mean() 

#How do I explore a pandas Series?
data.City.describe()
data.City.value_counts(normalize=True).head()
data.City.unique()

%matplotlib inline
data.City.count().plot(kind = 'hist')

#How do I handle missing values in pandas?
ufo = pd.read_csv("http://bit.ly/uforeports")
ufo.tail()
ufo.isnull().tail()
ufo.isnull().sum(axis=1)
pd.Series([True,False,True]).sum()
ufo[ufo.City.isnull()]
ufo.shape
ufo.dropna(how='any').shape
ufo.dropna(how='all').shape
ufo.dropna(subset = ['City','Shape Reported'],how = 'all').shape
ufo['City'].fillna(value = 'VARIOUS',inplace=True)
ufo['City'].value_counts(dropna=False)
ufo[ufo['City']=='VARIOUS'].value_counts()


#What do I need to know about the pandas index? (Part 1)
drinks = pd.read_csv("http://bit.ly/drinksbycountry")
drinks.tail()
drinks.index
drinks.columns
drinks.loc[23,'beer_servings']
drinks.set_index ('country',inplace = True)
drinks.head()
drinks.loc['Brazil','beer_servings']
drinks.describe().loc['50%','beer_servings']  
drinks.continent.head()
drinks.continent.value_counts().values
drinks.continent.value_counts()['Africa']
people = pd.Series([30000,43243],index = ['Albania','Zimbabwe'],name = 'Population')
people
drinks.beer_servings *people
pd.concat([drinks,people],axis =1 ).head()

drinks.reset_index(inplace =True)
#ow do I select multiple rows and columns from a pandas DataFrame?
drinks.loc[:,:] # it is for selecting rows and columns acc to labels
drinks.loc[0:2,'beer_servings':'wine_servings']
drinks[drinks['beer_servings']== 89].continent
drinks.iloc[:,0:-1] # it serves by integer position
drinks[['beer_servings','wine_servings']]

#When should i use inplace parameter in pandas

#How do I make my pandas DataFrame smaller and faster?
sorted(drinks.continent.unique())
drinks['continent'] = drinks.continent.astype('category')
drinks.dtypes
drinks.continent.cat.codes.head()
drinks.memory_usage(deep = True)
df = pd.DataFrame({'ID':[100,101,102,103],'quality':['Good','very Good','Better','excellent']})
df.head()
df.sort_values('quality')
df['quality'] = df.quality.astype('category',categories = ['Good','very Good', 'Better', 'excellent'],ordered = True)
df.loc[df.quality > 'Good']


#More of the questions answered
drinks.sample(n=5)
drinks.sample(frac = 0.75)
drinks['sex_male'] = drinks.sex.map({'female':0,'Male':1})

pd.to_datetime(ufo.Time)



def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)
        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.
    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    import warnings
    warnings.filterwarnings("ignore")
    graph = nx.Graph()
    screen_names = [user['screen_name'] for user in users]
    graph.add_nodes_from(screen_names)
    count = counter()
    users_friend = count.update({j:k for j,k in friends_counts.})
    graph.add_nodes_from(users_friend)
    """for i,j in friend_counts:
        if(j>1):
            graph.add_node(i)
            graph.add_edge(i,users"""
    nx.draw(graph, with_labels=True)
    return graph
%matplotlib inline
graph = create_graph(users,count.most_common())
print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))