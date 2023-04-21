import cv2
from deepface import DeepFace
import numpy as np
import pandas as pd
# Faker library is used to assign dummy names to face images
from faker import Faker
import os

justOnce = True

# model_name can take any values among ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
model_name = 'DeepFace'

# distance_metric can take any values among ["cosine", "euclidean", "euclidean_l2"]
distance_metric = 'cosine'

# detector_backend can take any values among ['opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'mediapipe']
detector_backend = 'opencv'

fake = Faker()

def getImage(frame, coordinates):
  x, y, w, h = coordinates.values()
  return frame[y:y+h, x:x+w]
 
cap = cv2.VideoCapture('classroom.mp4')
 
if (cap.isOpened() == False): 
  print("Error opening video stream or file")
 
while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
 
    cv2.imshow('Frame',frame)

    faces = DeepFace.extract_faces(img_path = frame, detector_backend = detector_backend, enforce_detection=False, align=True)
    print("face obj", faces)

    for face in faces:
      if justOnce:
        cv2.imwrite(f'database/{fake.name()}.png', getImage(frame, face['facial_area']))
        justOnce = False

      dfs = DeepFace.find(img_path = getImage(frame, face['facial_area']), 
        db_path = "database", 
        detector_backend = detector_backend,
        enforce_detection=False,
        distance_metric=distance_metric,
        model_name=model_name, 
        align=True
      )
      print('pandas df = ',dfs[0])
      # If any face Image from database matches 75% or more, then we dont write the newly found face to database, else we write it
      if dfs[0].empty == False and max(dfs[0]['DeepFace_cosine']) < 0.75:
        cv2.imwrite(f'database/{fake.name()}.png', getImage(frame, face['facial_area']))
        
        # Deleting the pickle file as new face images are added to DB, and indices need to be created again
        if os.path.exists("./database/representations_deepface.pkl"):
            os.remove("./database/representations_deepface.pkl")


    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
 
  else: 
    break
 
cap.release()
cv2.destroyAllWindows()