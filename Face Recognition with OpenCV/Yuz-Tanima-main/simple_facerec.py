import face_recognition
import cv2
import os
import glob
import numpy as np
#Gerekli kutuphaneleri kuruyorum.

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        # Encode edilmis resimleri yukletiyorum.

        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Kaydedilen resimler
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # İlk dosya ismini aliyorum.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Encode islemi yapiyorum.
            img_encoding = face_recognition.face_encodings(rgb_img)[0]


            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Tum resimlerden ve videolardan yuzleri tanimasini sagliyorum.
        # Resimleri BGR colordan(opencv icin) RGB colora ceviriyorum(face recognitionda tanimasi icin).
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Yuzleri tanirsa eslesme oluyor.
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # Tanima islemi olmazsa Unknown yazdiriyorum.

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Yuz tanimasi yapilacak yerin lokasyonunu ayarliyorum.
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names  #Yuzlerin lokasyonlari ve isimleri
