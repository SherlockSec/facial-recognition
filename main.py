import cv2;
import os;
import os.path;
import numpy as np;
import sys;

TFR_subjects = dict();

TFR_cascade_path = variable();


def TFR_face_detect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY);
    
    TFR_bool_cascade_check = os.path.isfile(TFR_cascade_path);

    if !TFR_bool_cascade_check:

        raise ValueError('Error: Cascade Sheet does not exist. Check the path set in your code.');

    else:    

        face_cascade = cv2.CascadeClassifier(TFR_cascade_path);

        TFR_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
        if (len(TFR_faces) == 0):
            return None, None;
            raise ValueError('Error: No faces detected.');

        if TFR_mode_select = 0:
            TFR_detection = TFR_single_face_detect();
            return TFR_detection;
        elif TFR_mode_select = 1:
            TFR_detection = TFR_multi_face_detect();
            return TFR_detection;
    
    
def TFR_single_face_detect(func_faces):
    (x, y, w, h) = faces[0];                      
    return gray[y:y+w, x:x+h], faces[0];          

def TFR_multi_face_detect(func_faces):
    TFR_counter = 0;
    for ( x, y, w, h ) in func_faces:                   
        ( x, y, w, h ) = func_faces[TFR_counter];
        return gray[y:y+w, x:x+h], func_faces[TFR_counter];
        TFR_counter += 1;   
    


def TFR_prepare_training_data(data_folder_path):
    
    dirs = os.listdir(data_folder_path);
    
    faces = [];
    labels = [];
    
    for dir_name in dirs:
        
        if not dir_name.startswith("s"):
            continue;
            
        label = int(dir_name.replace("s", ""));
        
        subject_dir_path = data_folder_path + "/" + dir_name;
        
        subject_images_names = os.listdir(subject_dir_path);
        
        for image_name in subject_images_names:
            
            if image_name.startswith("."):
                continue;
            
            image_path = subject_dir_path + "/" + image_name

            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(100)
            
            face, rect = detect_face(image)
            
            if face is not None:
                faces.append(face)
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels


print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)


def predict(test_img):
    img = test_img.copy()
    face, rect = detect_face(img)

    label, confidence = face_recognizer.predict(face)
    label_text = subjects[label]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img


print("Predicting images...")

test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")
test_img4 = cv2.imread("test-data/test4.jpg")

predicted_img1 = predict(test_img1)
predicted_img2 = predict(test_img2)
predicted_img3 = predict(test_img3)
predicted_img4 = predict(test_img4)
print("Prediction complete")

cv2.imshow(subjects[1], cv2.resize(predicted_img1, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[3], cv2.resize(predicted_img3, (400, 500)))
cv2.imshow(subjects[4], cv2.resize(predicted_img4, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()
