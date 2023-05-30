import cv2
import os
from skimage import data

sailboats = []
y = []
folder_dir = '/home/agnieszka/PycharmProjects/MachineLearningCourse/Lab_13/example/sailboat'
for images in os.listdir(folder_dir):
    if (images.endswith('.jpg')):
        nazwa = '/home/agnieszka/PycharmProjects/MachineLearningCourse/Lab_13/example/sailboat/'+str(images)
        image = cv2.imread(nazwa)
        sailboats.append(image)
        y.append('sailboat')

warships = []
y = []
folder_dir = '/home/agnieszka/PycharmProjects/MachineLearningCourse/Lab_13/example/warship'
for images_war in os.listdir(folder_dir):
    if (images_war.endswith('.jpg')):
        nazwa = '/home/agnieszka/PycharmProjects/MachineLearningCourse/Lab_13/example/warship/'+str(images_war)
        image_war = cv2.imread(nazwa)
        warships.append(image_war)
        y.append('warship')
        # cv2.imshow('picture', image_war)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

feature_detector_descriptor = cv2.AKAZE_create()

