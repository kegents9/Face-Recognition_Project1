import cv2
import face_recognition

# Load the main image
imgmain = face_recognition.load_image_file('D:/sideProject/Proj1_python/148292.jpg')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)
imgmain_resize = cv2.resize(imgmain, (440, 450))

# Load the test image
imgTest = face_recognition.load_image_file('D:/sideProject/Proj1_python/0121.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
imgTest_resize = cv2.resize(imgTest, (400, 500))

# Locate and encode the face in the main image
faceLoc = face_recognition.face_locations(imgmain_resize)[0]
encodeElon = face_recognition.face_encodings(imgmain_resize, [faceLoc])[0]
cv2.rectangle(imgmain_resize, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Locate and encode the face in the test image
faceLocTest = face_recognition.face_locations(imgTest_resize)[0]
encodeTest = face_recognition.face_encodings(imgTest_resize, [faceLocTest])[0]
cv2.rectangle(imgTest_resize, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compare the faces and calculate the distance
results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)

# Print the results and distance
print(results, faceDis)

# Display the test image with the results and distance
cv2.putText(imgTest_resize, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
cv2.imshow('Test Image', imgTest_resize)
cv2.waitKey(0)