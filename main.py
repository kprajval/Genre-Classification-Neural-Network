import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def determine_genre(row):
    if row['energy_%'] > 0.7 and row['danceability_%'] > 0.7:
        return 'Dance'
    elif row['acousticness_%'] > 0.5:
        return 'Acoustic'
    elif row['valence_%'] < 0.3:
        return 'Sad'
    elif row['energy_%'] > 0.5 and row['liveness_%'] > 0.4:
        return 'Live'
    else:
        return 'Other' 

def dataProcessing(FilePath):
    df = pd.read_csv(FilePath)
    
    df = df.drop(['artist_count', 'released_year', 'released_month', 'released_day',
                   'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
                   'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts',
                   'in_shazam_charts'], axis=1)

    df['genre'] = df.apply(determine_genre, axis=1)

    label_encoder = LabelEncoder()

    df['key'] = label_encoder.fit_transform(df['key'])
    df['artist(s)_name'] = label_encoder.fit_transform(df['artist(s)_name'])
    df['mode'] = label_encoder.fit_transform(df['mode'])
    df['genre'] = label_encoder.fit_transform(df['genre']) 

    columns_to_normalize = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                            'acousticness_%', 'instrumentalness_%', 'liveness_%', 
                            'speechiness_%']
    
    for column in columns_to_normalize:
        df[column] = (df[column] - df[column].mean()) / df[column].std()

    return df

df_processed = dataProcessing('SpotifyMostStreamedSongs.csv')

feature_columns = ['bpm', 'danceability_%', 'valence_%', 'energy_%', 
                   'acousticness_%', 'instrumentalness_%', 'liveness_%', 
                   'speechiness_%']
target_column = 'genre'

X = df_processed[feature_columns]
y = df_processed[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = keras.Sequential([
    layers.Input(shape=(X_train_scaled.shape[1],)),  
    layers.Dense(128, activation='relu'),  
    layers.Dense(64, activation='relu'),   
    layers.Dense(32, activation='relu'),   
    layers.Dense(len(df_processed['genre'].unique()), activation='sigmoid')  
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

predictions = model.predict(X_test_scaled)
predicted_classes = predictions.argmax(axis=1) 