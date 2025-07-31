# Program To Read video and Extract Frames

# Import
import cv2
import os

# Path to video file
vidObj = cv2.VideoCapture("/home/biosense-ms/Desktop/gitRepos/OAKLuxonis/green_ball/offline_video_samples/sample.mp4")

if not vidObj.isOpened():
    print("Error: Could not open video file. Check the path and file format.")
else:
    print("Video file opened successfully.")
    
# Used as counter variable
count = 0

# checks whether frames were extracted
success = 1

print(os.path.abspath("sample.mp4"))
print(os.path.exists("sample.mp4"))

while success:
    success, image = vidObj.read()
    #print("Success: ", success)
    if success and image is not None:
        #cv2.imwrite("frame%d.jpg" % count, image)
        cv2.imshow("Frame", image)
        count += 1
