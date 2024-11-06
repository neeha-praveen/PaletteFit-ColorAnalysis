import cv2
import numpy as np

def extract_skin_tone(image, face):
    (x, y, w, h) = face
    face_region = image[y:y+h, x:x+w]
    hsv_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70])   
    upper_skin = np.array([20, 150, 255])  
    skin_mask = cv2.inRange(hsv_face, lower_skin, upper_skin)
    skin_area = cv2.bitwise_and(face_region, face_region, mask=skin_mask)
    avg_skin_color = cv2.mean(skin_area, mask=skin_mask)
    skin_tone = classify_skin_tone(avg_skin_color)

    return skin_tone

def classify_skin_tone(avg_color):
    avg_b, avg_g, avg_r = avg_color[0], avg_color[1], avg_color[2]
    avg_rgb = (avg_r, avg_g, avg_b)
    brightness = np.mean(avg_rgb)
    if brightness < 100:
        return "Dark"
    elif brightness < 180:
        return "Medium"
    else:
        return "Fair"

def extract_eye_color(image, face):
    (x, y, w, h) = face
    eye_region = image[y:y+h, x:x+w]
    eye_region = eye_region[:h//3, :]  
    hsv_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_eye[:, :, 0]) 
    eye_color = classify_eye_color(avg_hue)

    return eye_color

def classify_eye_color(avg_hue):
    if avg_hue < 10 or avg_hue > 160:
        return "Brown"
    elif avg_hue < 25:
        return "Green"
    elif avg_hue < 45:
        return "Blue"
    else:
        return "Unknown"
    
def extract_hair_color(image, face):
    (x, y, w, h) = face
    hair_region = image[y-40:y, x:x+w]  
    hsv_hair = cv2.cvtColor(hair_region, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv_hair[:, :, 0])
    hair_color = classify_hair_color(avg_hue)

    return hair_color

def classify_hair_color(avg_hue):
    if avg_hue < 10 or avg_hue > 160:  
        return "Black/Brown"
    elif avg_hue < 25:  
        return "Blonde"
    elif avg_hue < 45:  
        return "Red"
    else:
        return "Unknown"

def determine_season(skin_tone, eye_color, hair_color):
    if skin_tone == "Fair" and (eye_color == "Blue" or eye_color == "Green") and (hair_color == "Blonde" or hair_color == "Light Brown"):
        return "Spring"
    elif skin_tone == "Fair" and eye_color in ["Blue", "Gray"] and hair_color in ["Ash Blonde", "Light Brown"]:
        return "Summer"
    elif skin_tone == "Medium" or skin_tone == "Dark" and eye_color in ["Brown", "Hazel", "Dark Green"] and hair_color in ["Brown", "Auburn", "Golden Blonde"]:
        return "Autumn"
    elif skin_tone == "Fair" and eye_color in ["Dark Brown", "Blue", "Green"] and hair_color in ["Dark Brown", "Black"]:
        return "Winter"
    return "Unclassified"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image = cv2.imread('C:/Users/neeha/OneDrive/Desktop/vscode/PaletteFit-ColorAnalysis/sample.jpg') # you can input your own path to image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
if len(faces) > 0:
    face = faces[0]
    skin_tone = extract_skin_tone(image, face)
    eye_color = extract_eye_color(image, face)
    hair_color = extract_hair_color(image, face)

season = determine_season(skin_tone, eye_color, hair_color)
print(f"The seasonal palette for this person is: {season}")