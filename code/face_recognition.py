import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
import torch.nn as nn
import os
import torchvision.models as models
from PIL import Image
import psutil
# set the database where deposits the pictures to training
database = {}
# absolute file path of filename
current_dir = os.path.dirname(os.path.abspath(__file__))
# the path to the directory one level up
parent_dir = os.path.dirname(current_dir)
# the filename path
cascade_path = os.path.join(parent_dir, 'face_cascade', 'haarcascade_frontalface_default.xml')
# Load the face cascade
face_cascade = cv2.CascadeClassifier(cascade_path)
# preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.ToPILImage(),
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# face recognize
def Face_recognition(image):
    # 加载图像
    if image is None:
        print("filtpath wrong.")
    else:
        # RGB transform
        test_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # check weather correct
        if test_image.shape[-1] > 3:
            print("channel over 3.")


    # use face cascade to check face weather existing
    #gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    face_locations = face_cascade.detectMultiScale(test_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(face_locations) == 0:
        # no
        return None

    # yes,feature extract feature
    face_encodings = []
    for (x, y, w, h) in face_locations:
        face_roi = image[y:y+h, x:x+w]


        
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)


        # use model to extract feature
        with torch.no_grad():
            features = model(face_tensor).squeeze().cpu().numpy()
        face_encodings.append(features)

    # caculate the similarity between the image to be recognized and features of database
    similarities = []
    for encoding in face_encodings:
        max_similarity = -1  #
        # cacukate the similarity
        for person in database.database:
            person_features = person['features']
            for features in person_features:
                similarity = np.dot(encoding, features) / (np.linalg.norm(encoding) * np.linalg.norm(features))
                if similarity > max_similarity:
                    max_similarity = similarity
                    
            similarities.append(max_similarity)
            max_similarity = -1  
            
    # order by similarity
    sorted_indices = np.argsort(similarities)[::-1]  
        # return the label
    most_similar_index = sorted_indices[0]
    most_similar_label = database.database_labels[most_similar_index ]
    return most_similar_label

class FaceDatabase:
    def __init__(self):
        self.database = []
        self.database_labels = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = torch.nn.Linear(512, dataset_num)  
        self.resnet.to(self.device)
        self.resnet.eval()

    def build_database_from_dataset(self, dataset_path):
        # traverse the file
        for label in os.listdir(dataset_path):
            label_path = os.path.join(dataset_path, label)
            if os.path.isdir(label_path):
                
                features_list = []  #to stockpile the features
                # traverse the file of label
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)

                    # load image
                    image = cv2.imread(image_path)
                    if image is None:
                        print("Failed to read image:", image_path)
                    # detect faces
                    faces=self.detect_faces(image)
                    #extract the features to list
                    for face in faces:                     
                        features = self.extract_face_features(image, face)
                        features_list.append(features)
                
                if len(features_list) > 0:
                    # set thresholds
                    threshold = 0.0 
                    # build dictionary
                    person = {
                        'label': label,
                        'features': features_list,
                        'threshold': threshold
                    }

                    #add to database
                    self.database.append(person)
                    self.database_labels.append(label)
                    print(f"successfully add：{label}")
        print("database finish.")
    def detect_faces(self, image):
        if image is None:
            print("Image is empty.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #ensure the position of face
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def extract_face_features(self, image, face):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if isinstance(face, np.ndarray) and face.size > 0:
            face = face.astype(int)
        else:
            return None
        if face is None:
            return None
        x, y, width, height = list(face)
        face_image = image[y:y+height, x:x+width]
        face_image = transform(face_image).unsqueeze(0).to(self.device)

        # extract features by resnet
        with torch.no_grad():
            features = self.resnet(face_image).cpu().squeeze().numpy()

        return features

current_folder = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(current_folder, 'model_train.py')

if os.path.exists(train_path):
    with open(train_path, 'r', encoding='utf-8') as script_file:
        script_code = script_file.read()
        exec(script_code)
else:
    print(f"wrong：cant find {train_path}。")
os.system('cls' if os.name == 'nt' else 'clear')
#dataset and database
current_folder = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_folder, '..\\dataset')
dataset_num = len(next(os.walk(dataset_path))[1])
database = FaceDatabase()
database.build_database_from_dataset(dataset_path) 
os.system('cls' if os.name == 'nt' else 'clear') 
# model to feature extract
model = models.resnet18(pretrained=True)
num_classes = len(database.database_labels)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
current_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_folder, '..\\model')
model_file_path = os.path.join(model_folder, 'resnet18.pth')
model_path = os.path.abspath(model_file_path)
model.load_state_dict(torch.load(model_path))
# GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
#load dataset
def load_dataset(dataset_path):
    dataset = []
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                dataset.append((image_path, person_folder))
    return dataset

#add face to database by file
def add(add_path, data_path):
    label = input("label：")
    # check if existing
    label_path = os.path.join(data_path, label)
    if os.path.exists(label_path):
        print("label existed.")

        max_number = 0
        for filename in os.listdir(label_path):
            if filename.endswith(".jpg"):
                file_number = int(filename.split("_")[1].split(".")[0])
                max_number = max(max_number, file_number)
        # add to the old file
        new_filename = label + "_" + str(max_number + 1).zfill(3) + ".jpg"
    else:
        # new file
        os.makedirs(label_path)
        print("file build.")

        # new filename
        new_filename = label + "_001.jpg"
    new_filepath = os.path.join(label_path, new_filename)
    if add_path!=0:
        image = Image.open(add_path)
        # save in jpg
        image.save(new_filepath, "JPEG")

        print("successfully add to database！")
    else:

        capture_and_save_image(label,label_path)

 #add face by camera   
def capture_and_save_image(label,label_path):
    camera = cv2.VideoCapture(0)  
    if not camera.isOpened():
        print("cant open camera")
        return
    while True:
        ret, frame = camera.read()
        cv2.putText(frame, "Press 'p' to capture, press 'esc' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Camera", frame)
        #'esc' to break
        if cv2.waitKey(1) == 27:
            break

        # 'p' to screenshot
        if cv2.waitKey(1) == ord('p'):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                face = frame[y:y+h, x:x+w]

                # resize the face
                resized_face = cv2.resize(face, (250, 250))
                max_number = 0
                for filename in os.listdir(label_path):
                    if filename.endswith(".jpg"):
                        file_number = int(filename.split("_")[1].split(".")[0])
                        max_number = max(max_number, file_number)

                new_filename= label + "_" + str(max_number + 1).zfill(3) + ".jpg"
                data_path = os.path.join(label_path, new_filename)
                print(data_path)
                cv2.imwrite(data_path, resized_face)

                print("successfully.")
    # close camera
    camera.release()
    cv2.destroyAllWindows()
#train the feature extract model
def train_again(train_path):
    a=0
    if os.path.exists(train_path):
    # use exec to run the script
        with open(train_path, 'r', encoding='utf-8') as script_file:
            script_code = script_file.read()
            exec(script_code)
        a=1
    else:
        print(f"wrong:cant find {train_path}。")
        a=0
    return a

def file_face_recognition():
    image_path = input("picture path: ")
    test_image = cv2.imread(image_path)
    recognized_label = Face_recognition(test_image)
    if recognized_label is not None:
        print("result:", recognized_label)
    else:
        print("cant recognize")

def capture_face_recognition():
    # 打开摄像头
    video_capture = cv2.VideoCapture(0)
    
    try:
        while True:
            # 读取摄像头的帧
            ret, frame = video_capture.read()

            # 将帧转换为灰度图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 人脸检测
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # 在图像上绘制方框和类别名
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                recognized_label = Face_recognition(frame)
                if recognized_label is not None:
                    cv2.putText(frame, recognized_label, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 显示结果图像
            cv2.imshow('Video', frame)

            # 检查摄像头资源情况
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 获取当前摄像头的资源使用情况
            used_percentage = psutil.virtual_memory().percent

            # 设置资源占用阈值
            threshold_percentage = 80

            # 如果摄像头资源占用超过阈值，自动关闭摄像头
            if used_percentage > threshold_percentage:
                print("摄像头资源占用过多，自动关闭摄像头")
                # 释放摄像头和关闭窗口
                video_capture.release()
                cv2.destroyAllWindows()
                break
            

    except Exception as e:
        print("识别结果:", recognized_label)
        print(f"Error: {e}")

    finally:
        # 释放摄像头和关闭窗口
        video_capture.release()
        cv2.destroyAllWindows()
            
def main():
    while True:
        choice=int(input("choose：\n1. face recognize\n2. add face\n3. over\n"))
        os.system('cls' if os.name == 'nt' else 'clear')
        if choice==1:
            while True:
                a = int(input("choose the way to recognize：\n1. file path\n2. camera\n3. stop\n"))
                os.system('cls' if os.name == 'nt' else 'clear')
                if a==1:
                    file_face_recognition()
                elif a==2:
                    capture_face_recognition()
                elif a==3:
                    break
                else:
                    print("repeat!")
                
        elif choice==2:
            while True:
                b = int(input("the way to add：\n1. filepath \n2. camera\n"))
                os.system('cls' if os.name == 'nt' else 'clear')
                if b == 1:
                    add_path = input("file path：")
                    add(add_path, dataset_path)
                    if train_again(train_path)==1:
                        dataset_num = len(next(os.walk(dataset_path))[1])
                        database={}
                        database = FaceDatabase()
                        database.build_database_from_dataset(dataset_path)
                        break
                    else:
                        print("check train model file!\n")
                        break
                elif b == 2:
                    add_path=0
                    add(add_path, dataset_path)
                    if train_again(train_path)==1:
                        dataset_num = len(next(os.walk(dataset_path))[1])
                        database={}
                        database = FaceDatabase()
                        database.build_database_from_dataset(dataset_path)
                        break
                    else:
                        print("check train model file!\n")
                        break
                else:
                    print("input wrong!repeat!\n")

            
        elif choice==3:
            break
        else:
            print("repeat:")  
if __name__ == "__main__":
    main()