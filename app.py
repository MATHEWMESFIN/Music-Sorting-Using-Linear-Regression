import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title='Music Sorting Using Linear Regression', page_icon='🎵', layout='wide')
st.title('Music Sorting Using Linear Regression')

tracksData = pd.read_csv('cleaned_spotify_tracks_data.csv')

# To predict the danceability of a track, we will use a linear regression model
# we will use the following features: energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo
# we will use the following target: danceability

X = tracksData[['energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
y = tracksData['danceability']

# display the metrics for the danceability level
col1, col2 = st.columns(2)
with col1:
    st.header('Danceability Level Value Counts')
    st.metric('Non-dance', tracksData['danceability_level'].value_counts().sort_index()[0])
    st.metric('Light Dance', tracksData['danceability_level'].value_counts().sort_index()[1])
    st.metric('Hard Dance', tracksData['danceability_level'].value_counts().sort_index()[2])

with col2:
    st.header('Correlation Analysis')
    st.write(tracksData.corr()['danceability'])

energyDanceability = tracksData.groupby(['danceability_level', tracksData['energy'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
energyDanceability = energyDanceability.pivot(index='energy', columns='danceability_level', values='count').fillna(0)
energyDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Energy vs Danceability')
energyCol1, energyCol2 = st.columns([2, 1])
energyCol1.line_chart(energyDanceability)
with energyCol2:
    st.subheader('Observations')
    st.caption(' - Most hard dance tracks have energy levels around 0.7.')
    st.caption(' - Most light dance tracks have energy levels around 0.9.')
    st.caption(' - Most non-dance tracks are split between 0.1 and 0.95.')


loudnessDanceability = tracksData.groupby(['danceability_level', tracksData['loudness'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
loudnessDanceability = loudnessDanceability.pivot(index='loudness', columns='danceability_level', values='count').fillna(0)
loudnessDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Loudness vs Danceability')
loudnessCol1, loudnessCol2 = st.columns([2, 1])
loudnessCol1.line_chart(loudnessDanceability)
with loudnessCol2:
    st.subheader('Observations')
    st.caption(' - Mostly all tracks have loudness levels between -20 and -2.')

speechinessDanceability = tracksData.groupby(['danceability_level', tracksData['speechiness'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
speechinessDanceability = speechinessDanceability.pivot(index='speechiness', columns='danceability_level', values='count').fillna(0)
speechinessDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Speechiness vs Danceability')
speechCol1, speechCol2 = st.columns([2, 1])
speechCol1.line_chart(speechinessDanceability)
with speechCol2:
    st.subheader('Observations')
    st.caption(' - There are not many tracks with high speechiness.')

acousticnessDanceability = tracksData.groupby(['danceability_level', tracksData['acousticness'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
acousticnessDanceability = acousticnessDanceability.pivot(index='acousticness', columns='danceability_level', values='count').fillna(0)
acousticnessDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Acousticness vs Danceability')
acousticCol1, acousticCol2 = st.columns([2, 1])
acousticCol1.line_chart(acousticnessDanceability)
with acousticCol2:
    st.subheader('Observations')
    st.caption(' - Magority of all tracks have acousticness levels less than 0.1.')

instrumentalnessDanceability = tracksData.groupby(['danceability_level', tracksData['instrumentalness'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
instrumentalnessDanceability = instrumentalnessDanceability.pivot(index='instrumentalness', columns='danceability_level', values='count').fillna(0)
instrumentalnessDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Instrumentalness vs Danceability')
instrumentCol1, instrumentCol2 = st.columns([2, 1])
instrumentCol1.line_chart(instrumentalnessDanceability)
with instrumentCol2:
    st.subheader('Observations')
    st.caption(' - Magority of all tracks have instrumentalness levels less than 0.1.')

livenessDanceability = tracksData.groupby(['danceability_level', tracksData['liveness'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
livenessDanceability = livenessDanceability.pivot(index='liveness', columns='danceability_level', values='count').fillna(0)
livenessDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Liveness vs Danceability')
liveCol1, liveCol2 = st.columns([2, 1])
liveCol1.line_chart(livenessDanceability)
with liveCol2:
    st.subheader('Observations')
    st.caption(' - Most tracks have a liveness level around 0.1.')

valenceDanceability = tracksData.groupby(['danceability_level', tracksData['valence'].apply(lambda x: round(x, 1))]).size().reset_index().rename(columns={0: 'count'})
valenceDanceability = valenceDanceability.pivot(index='valence', columns='danceability_level', values='count').fillna(0)
valenceDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Valence vs Danceability')
valenceCol1, valenceCol2 = st.columns([2, 1])
valenceCol1.line_chart(valenceDanceability)
with valenceCol2:
    st.subheader('Observations')
    st.caption(' - Most light dance tracks have valence levels between 0.2 and 0.4.')
    st.caption(' - Most hard dance tracks have valence levels between 0.6 and 0.9.')
    st.caption(' - Most non-dance tracks have valence levels between 0 and 0.15.')

tempoDanceability = tracksData.groupby(['danceability_level', tracksData['tempo'].apply(lambda x: round(x, -1))]).size().reset_index().rename(columns={0: 'count'})
tempoDanceability = tempoDanceability.pivot(index='tempo', columns='danceability_level', values='count').fillna(0)
tempoDanceability.columns = ['Non-dance', 'Light Dance', 'Hard Dance']

st.header('Tempo vs Danceability')
tempoCol1, tempoCol2 = st.columns([2, 1])
tempoCol1.line_chart(tempoDanceability)
with tempoCol2:
    st.subheader('Observations')
    st.caption(' - Most tracks have tempo levels between 70 and 180.') 
    st.caption(' - Most hard dance tracks have tempo levels between 115 and 125.')
    st.caption(' - Most light dance tracks have tempo levels between 125 and 140.')
    st.caption(' - Non-dance tracks are not clustered in a specific tempo range.')


# get variables for trining and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.header('Linear Regression Model')

# create a linear regression model
model = LinearRegression()

# train the model
model.fit(X_train, y_train)

# compare the predicted danceability with the actual danceability
predictedVsActual = pd.DataFrame()
predictedVsActual['danceability_pred'] = model.predict(X_test)
predictedVsActual['danceability'] = y_test.values

# remove any values that are less than 0 or greater than 1
predictedVsActual = predictedVsActual[predictedVsActual['danceability_pred'] >= 0]
predictedVsActual = predictedVsActual[predictedVsActual['danceability_pred'] <= 1]

# categorize the values of danceability into 3 categories
# Non-dance: 0.0 - 0.33
# Light dance: 0.34 - 0.66
# Hard Dance: 0.67 - 1.0

# add columns for the danceability levels
predictedVsActual['danceability_pred_level'] = pd.cut(predictedVsActual['danceability_pred'], bins=[0.0, 0.33, 0.66, 1.0], labels=[0, 1, 2])
predictedVsActual['danceability_level'] = pd.cut(predictedVsActual['danceability'], bins=[-0.1, 0.33, 0.66, 1.0], labels=[0, 1, 2])

# display metrics for the predicted danceability level
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Predicted Non-dance', predictedVsActual['danceability_pred_level'].value_counts().sort_index()[0])
    st.metric('Actual Non-dance', predictedVsActual['danceability_level'].value_counts().sort_index()[0])
with col2:
    st.metric('Predicted Light Dance', predictedVsActual['danceability_pred_level'].value_counts().sort_index()[1])
    st.metric('Actual Light Dance', predictedVsActual['danceability_level'].value_counts().sort_index()[1])
with col3:
    st.metric('Predicted Hard Dance', predictedVsActual['danceability_pred_level'].value_counts().sort_index()[2])
    st.metric('Actual Hard Dance', predictedVsActual['danceability_level'].value_counts().sort_index()[2])


st.header('Predicted vs Actual Danceability')
resultsCol1, resultsCol2 = st.columns(2)
with resultsCol1:
    st.write(predictedVsActual)

with resultsCol2:
    # get the mse of the model
    mse = np.mean((model.predict(X_test) - y_test) ** 2)
    st.metric('Mean Squared Error', mse)

    # get the rmse of the model
    rmse = np.sqrt(mse)
    st.metric('Root Mean Squared Error', rmse)

    # get the r2 score of the model
    r2 = model.score(X_test, y_test)
    st.metric('R2 Score', r2)