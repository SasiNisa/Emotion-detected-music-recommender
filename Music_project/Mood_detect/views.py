from django.shortcuts import render
import os
from django.conf import settings

# Create your views here.
from .form import ImageForm
from .models import Image

from keras.models import load_model
from time import sleep
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import cv2
from googleapiclient.discovery import build
import pandas as pd
import numpy as np

#display(Play)
# Making Songs Recommendations Based on Predicted Class
def Recommend_Songs(pred_class):
    Music_Player = pd.read_csv("data_moods.csv")
    Music_Player = Music_Player[['name','artist','mood','popularity']]
    Play = Music_Player[Music_Player['mood'] =='Calm' ]
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    
    if( pred_class=='Disgust' ):

        Play = Music_Player[Music_Player['mood'] =='Sad' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)

    if( pred_class=='Happy' or pred_class=='Sad' ):

        Play = Music_Player[Music_Player['mood'] =='Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)

    if( pred_class=='Fear' or pred_class=='Angry' ):

        Play = Music_Player[Music_Player['mood'] =='Calm' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)

    if( pred_class=='Surprise' or pred_class=='Neutral' ):

        Play = Music_Player[Music_Player['mood'] =='Energetic' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        print(Play)

    print(Play)  
    
    return Play


#to do

""" Set up the YouTube API:

Go to the Google Developers Console (https://console.developers.google.com/).
Create a new project.
Enable the YouTube Data API v3 for your project.
Create credentials to get the API key.
Update  your Django view function (views.py) to include the logic for generating links to play the songs:"""

# end to do


# Assuming you have already defined the detect_emotion and Recommend_Songs functions here

def get_youtube_links(song_title, artist_name, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    search_query = f"{song_title} {artist_name}"
    request = youtube.search().list(q=search_query, type='video', part='id', maxResults=1)
    response = request.execute()
    if 'items' in response:
        video_id = response['items'][0]['id']['videoId']
        return f"https://www.youtube.com/watch?v={video_id}"
    else:
        return None



def detect_emotion(image_path):
    face_classifier = cv2.CascadeClassifier(r'C:\Users\User\Desktop\Mood-detection\Music_project\haarcascade_frontalface_default.xml')
    classifier = load_model(r"C:\Users\User\Desktop\Mood-detection\Music_project\model.h5")

    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

    frame = cv2.imread(image_path)

    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    # Show the uploaded image for debugging
    #cv2.imshow('Uploaded Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            return label

    return 'No Faces'



def index(request):
    img = Image.objects.all()
    recommended_songs_dict = []
    if request.method == "POST":
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance
            emotion = detect_emotion(obj.image.path)
            pred_class = emotion
            recommended_songs = Recommend_Songs(pred_class)
            if recommended_songs is None or recommended_songs.empty:
                # Handle the case when no recommendations are found
                recommended_songs_dict = []
            else:
                # Convert the DataFrame to a list of dictionaries
                recommended_songs_dict = recommended_songs.to_dict(orient='records')
                # Add YouTube links to each song in the recommended list
                # api_key = 'AIzaSyAs_vNpKm-i2P3uNxz0vfM5YeGecI6hX4U'
                # for song in recommended_songs_dict:
                #     youtube_link = get_youtube_links(song['name'], song['artist'], api_key)
                #     song['youtube_link'] = youtube_link
            return render(request, "index.html", {"obj": obj, "emotion": emotion, "recommended_songs":recommended_songs_dict})
    else:
        form = ImageForm()
        img = Image.objects.all()
    return render(request, "index.html", {"img": img, "form": form})
