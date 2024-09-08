import face_recognition
import os, sys
import math
import cv2
import numpy as np
import time
# import mysql.connector
# # from mysql.connector import connect, Error
# from getpass import getpass
# import datetime

def face_confidence(face_distance, threshold = 0.6):
    range = (1.0-threshold)
    linear_val = (1.0-face_distance)/(range*2.0)

    if face_distance > threshold:
        return str(round(linear_val*100, 2))+'%'
    else:
        value = (linear_val + ((1.0-linear_val)*math.pow((linear_val-0.5)*2, 0.2)))*100
        return str(round(value, 2))+'%'
    
class FaceRecognition:
    known_faces_encodings = []
    known_names = []
    process_current_frame = True
    face_locations = []
    face_encodings = []
    face_names = []

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('face_system/faces'):
            face_image = face_recognition.load_image_file(f'face_system/faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_faces_encodings.append(face_encoding)
            self.known_names.append(image)
        #print(self.known_names)

    def attendance(name):
            today = time.strftime('%d_%m_%Y')
            f = open(f'face_system/Records/record_{today}.csv','a')
            f.close()

            with open(f'face_system/Records/record_{today}.csv','r') as f:
                data = f.readlines()
                names = []
                for line in data:
                    entry = line.split(" ")
                    names.append(entry[0])
                
            with open(f'face_system/Records/record_{today}.csv','a') as f:
                if name not in names:
                    current_time = time.strftime('%H:%M:%S')
                    f.write(f"\n{name},{current_time}")

    def run_recognition(self):

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                
                small_frame = cv2.resize(frame, (0,0), fx = 0.25, fy=0.25)
                #img_rgb_small = small_frame[:,:,::-1]
                img_rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                #find all faces in the frame
                self.face_locations = face_recognition.face_locations(img_rgb_small)
                self.face_encodings = face_recognition.face_encodings(img_rgb_small, self.face_locations)

                self.face_names = []
                
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_faces_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_faces_encodings, face_encoding)
                    best_match = np.argmin(face_distances)

                    if matches[best_match]:
                        name = self.known_names[best_match]
                        confidence = face_confidence(face_distances[best_match])

                    self.face_names.append(f'{name} ({confidence})')
            
            self.process_current_frame = not self.process_current_frame

            for(top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom),(0,0,255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom),(0,0,255), -1)
                cv2.putText(frame, name, (left+6, bottom-6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 1)
                FaceRecognition.attendance(name)

            time.sleep(0.04)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1)==ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

# class Database:
#     connection = None
#     def __init__(self):
#         if self.connection is None:
#             self.connection = mysql.connector.connect(
#             host='localhost',
#             password=getpass("Enter password: "),
#             user='root',
#             database = 'People'
#         )
#         print(self.connection)
            
#     def find_data(self,file_name):
#         query = 'SELECT * FROM PEOPLE WHERE PICTURE=="{}"'.format(file_name)
#         with self.connection.cursor() as cursor:
#             cursor.execute(query)
#             print(cursor.fetchall())
    
#     def submit_data(self,name,dob):
#         query = 'ALTER TABLE PEOPLE ADD VALUES({0},{1})'.format(name,dob)
#         with self.connection.cursor() as cursor:
#             cursor.execute(query)

if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
    #print(fr.known_names,fr.known_faces_encodings)
    # while(True):
        
        # print("---MENU---")
        # print("1. Recognize face")
        # print("2. Submit Face")
        # print("3. Quit")
        # choice = int(input("Choice: "))
        # if(choice == 1):
        #     fr.run_recognition()
        #     # db = Database()
        #     # db.find_data()
        # elif(choice == 2):
        #     #fr = FaceRecognition()
        #     fr.save_image()
        #     # db = Database()
        #     # db.submit_data()
        # else:
        #     break;
