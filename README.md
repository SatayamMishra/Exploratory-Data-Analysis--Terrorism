# Name: Satyam mishra

# Task 4: Exploratory Data Analysis- Terrorism

# GRIP@TheSparkFoundation- Data Science and Buisness Analytics- june2022

# Objective

● Perform ‘Exploratory Data Analysis’ on dataset ‘Global Terrorism’

● As a Security/Defence analyst, try to find out the hot zone of terrorism.

● What all security issues and insights you can derive by EDA?

# Import all required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from wordcloud import WordCloud
import seaborn as sns

# Reading the data

dataset=pd.read_csv("C:/csvfile/globalterrorismdb_0718dist.csv",encoding='ISO-8859-1')
print("Data Import Successfully")
dataset.head(10)

dataset.info()

dataset.describe()

dataset.shape

# Checking missing value

dataset.isnull().sum()

# Checking the duplicates value

dataset.duplicated().sum()

dataset.nunique()

dataset.columns

# Rename the columns

dataset.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','provstate':'state',
                       'region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed',
                       'nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type',
                       'weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

dataset=dataset[['Year','Month','Day','Country','state','Region','city','latitude','longitude','AttackType','Killed',
               'Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

dataset["Country"].unique()

dataset["AttackType"].unique()

dataset["Motive"].unique()

dataset["Weapon_type"].unique()

# Destructive Features of Data

print("Country with the most attacks:",dataset['Country'].value_counts().idxmax())
print("City with the most attacks:",dataset['city'].value_counts().index[1]) 
print("Region with the most attacks:",dataset['Region'].value_counts().idxmax())
print("Year with the most attacks:",dataset['Year'].value_counts().idxmax())
print("Month with the most attacks:",dataset['Month'].value_counts().idxmax())
print("Group with the most attacks:",dataset['Group'].value_counts().index[1])
print("Most Attack Types:",dataset['AttackType'].value_counts().idxmax())

cities = dataset.state.dropna(False)
plt.subplots(figsize=(10,8))
wordcloud = WordCloud(background_color = 'white',
                     width = 512,
                     height = 384).generate(' '.join(cities))
plt.axis('off')
plt.imshow(wordcloud)
plt.show()

# Data Visualization

plt.hist(dataset['Country'],color="green")
plt.xlabel("Country")
plt.show()

plt.hist(dataset['AttackType'],color="blue")
plt.xlabel("AttackType")
plt.show()

plt.hist(dataset['Weapon_type'],color="orange")
plt.xlabel("Weapon_type")
plt.show()

# plt.show()

plt.figure(figsize=(12,8))
plt.plot(dataset['AttackType'])
plt.show()

dataset['Year'].value_counts(dropna = False).sort_index()

x_year = dataset['Year'].unique()
y_count_years = dataset['Year'].value_counts(dropna = False).sort_index()
plt.figure(figsize = (14,12))
sns.barplot(x = x_year,
           y = y_count_years,
           palette = 'rocket')
plt.xticks(rotation = 45)
plt.xlabel('Attack Year')
plt.ylabel('Number of Attacks each year')
plt.title('Attack_of_Years')
plt.show()

dataset.plot(kind ="scatter", 
          x ='Weapon_type', 
          y ='AttackType') 
plt.grid()

plt.subplots(figsize=(15,10))
sns.countplot('Year',data=dataset,palette='RdYlGn_r',edgecolor=sns.color_palette("YlOrBr", 10))
plt.xticks(rotation=45)
plt.title('Number Of Terrorist Activities Each Year',color="green",fontsize=15)
plt.show()

pd.crosstab(dataset.Year, dataset.Region).plot(kind='area',figsize=(12,8))
plt.title('Terrorist Activities by Region in each Year',color="green",fontsize=15)
plt.ylabel('Number of Attacks')
plt.show()

dataset['Wounded'] = dataset['Wounded'].fillna(0).astype(int)
dataset['Killed'] = dataset['Killed'].fillna(0).astype(int)
dataset['casualities'] =dataset['Killed'] + dataset['Wounded']

# Values are sorted by the top 40 worst terror attacks as to keep the heatmap simple and easy to visualize

terror=dataset.sort_values(by='casualities',ascending=False)[:40]

heat=terror.pivot_table(index='Country',columns='Year',values='casualities')
heat.fillna(0,inplace=True)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
colorscale = [[0, '#edf8fb'], [.3, '#00BFFF'],  [.6, '#8856a7'],  [1, '#810f7c']]
heatmap = go.Heatmap(z=heat.values, x=heat.columns, y=heat.index, colorscale=colorscale)
data = [heatmap]
layout = go.Layout(
    title='Top 40 Worst Terror Attacks in History from 1982 to 2016',
    xaxis = dict(ticks='', nticks=20),
    yaxis = dict(ticks='')
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='heatmap',show_link=False)

dataset.Country.value_counts()[:15]

# Top Countries affected by Terror Attacks

plt.subplots(figsize=(12,8))
sns.barplot(terror['Country'].value_counts()[:15].index,terror['Country'].value_counts()[:15].values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 90)
plt.show()

# Terrorist Attacks on a particular year and their Locations

import folium
from folium.plugins import MarkerCluster 
filterYear = terror['Year'] == 2008

filterData = terror[filterYear] 
reqFilterData = filterData.loc[:,'city':'longitude'] 
reqFilterData = reqFilterData.dropna() 
reqFilterDataList = reqFilterData.values.tolist()

map = folium.Map(location = [0, 30], tiles='CartoDB positron', zoom_start=2)
markerCluster = folium.plugins.MarkerCluster().add_to(map)
for point in range(0, len(reqFilterDataList)):
    folium.Marker(location=[reqFilterDataList[point][1],reqFilterDataList[point][2]],
                  popup = reqFilterDataList[point][0]).add_to(markerCluster)
map

terror.Group.value_counts()[1:15]

killData = terror.loc[:,'Killed']
print('Number of people killed by terror attack:', int(sum(killData.dropna())))

attackData = terror.loc[:,'AttackType']
typeKillData = pd.concat([attackData, killData], axis=1)

typeKillFormatData = typeKillData.pivot_table(columns='AttackType', values='Killed', aggfunc='sum')
typeKillFormatData


countryData = terror.loc[:,'Country']
countryKillData = pd.concat([countryData, killData], axis=1)

countryKillFormatData = countryKillData.pivot_table(columns='Country', values='Killed', aggfunc='sum')
countryKillFormatData

labels = countryKillFormatData.columns.tolist()
labels = labels[:50] 
index = np.arange(len(labels))
transpoze = countryKillFormatData.T
values = transpoze.values.tolist()
values = values[:50]
values = [int(i[0]) for i in values] 
colors = ['red', 'green', 'blue', 'purple', 'yellow', 'brown', 'black', 'gray', 'magenta', 'orange']  
fig, ax = plt.subplots(1, 1)
ax.yaxis.grid(True)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=25
fig_size[1]=25
plt.rcParams["figure.figsize"] = fig_size
plt.bar(index, values, color = colors, width = 0.9)
plt.ylabel('Killed People', fontsize=20)
plt.xlabel('Countries', fontsize = 20)
plt.xticks(index, labels, fontsize=18, rotation=90)
plt.title('Number of people killed by countries', fontsize = 20)

plt.show()


# Conclusion

### From the above graphs we can understand the number of terrorist attacks that happened all over the world and number of peoplekilled in each terrorist attacks.The Middle East and North Africa are seen to be the places of serious terrorist attacks.In addition, even though there is a perception that Muslims are supporters of terrorism, Muslims are the people who are most damaged by terrorist attacks. If you look at the graphics, it appears that Iraq, Afghanistan and Pakistan are the most damaged countries. All of these countries are Muslim countries.
