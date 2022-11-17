import cv2
import simple_facerec

# Taninacak gorselleri klasore ekliyorum --- openpyxl setupla
sfr = simple_facerec.SimpleFacerec()
sfr.load_encoding_images("images/")

# Kamerami seciyorum.
cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    # Yuz Tanima Kismi
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2) #Adini yazdiriyorum.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4) #Ekranda dikdortgen olustuyorum.

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break #Esc'ye basana kadar dongu devam ediyor.

cap.release()
cv2.destroyAllWindows()
