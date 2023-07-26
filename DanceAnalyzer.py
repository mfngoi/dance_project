

'''
1)  import neccessary libraries
2)  select dance video to analyze
        - video can be prerecorded or stream webcam
3)  make sure to set up mediapipe tools

4) check to see if video opened successfully

5) in a while loop start reading each frame of the video
    For every frame:
        - resize the image
        - use mediapipe to analyze image to get results and draw landmarks
        - draw angles onto the joints
        - display the image

6) release the video object


overlay1 = overlay()
overlay1.displayUI()

'''

import cv2 
import mediapipe as mp
import numpy as np
from utilities import save_file, resize_img
from overlay import overlay

class danceanalyzer:
    
    # The class will set up the mediapipe tools to be used in later functions
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()

    def analyzepic(self, image):

        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imageRGB)                   
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return image, results
    
    def getBodyPlace(self, image, results, bodyPart):

        height = image.shape[0]
        width = image.shape[1]

        x_place = int((1-results.pose_landmarks.landmark[bodyPart].x) * width)
        y_place = int(results.pose_landmarks.landmark[bodyPart].y * height)

        return (x_place, y_place)
        
    def getBodyAngle(self, results, body1, body2, body3): 
        x_body1 = results.pose_landmarks.landmark[body1].x 
        y_body1 = results.pose_landmarks.landmark[body1].y 

        x_body2 = results.pose_landmarks.landmark[body2].x
        y_body2 = results.pose_landmarks.landmark[body2].y

        x_body3 = results.pose_landmarks.landmark[body3].x
        y_body3 = results.pose_landmarks.landmark[body3].y

        radians = (np.arctan2((y_body3 - y_body2),(x_body3 - x_body2))) - (np.arctan2((y_body1 - y_body2),(x_body1 - x_body2)))
        degree = np.abs(radians * 180 / np.pi)

        if degree > 180.0:
            degree = 360 - degree

        return round(degree)

    def drawAllAngles(self, image, results):
        angleList = []
        right_elbow = self.getBodyAngle(results, 12, 14, 16)
        angleList.append(right_elbow)
        cv2.putText(image, str(right_elbow), self.getBodyPlace(image, results, 14), 1, 2, (0,0,255)) # Right Elbow [0]
        
        left_elbow = self.getBodyAngle(results, 11, 13, 15)
        angleList.append(left_elbow)
        cv2.putText(image, str(left_elbow), self.getBodyPlace(image, results, 13), 1, 2, (0,0,255)) # Left Elbow [1]
        
        right_knee = self.getBodyAngle(results, 24, 26, 28)
        angleList.append(right_knee)
        cv2.putText(image, str(right_knee), self.getBodyPlace(image, results, 26), 1, 2, (0,0,255)) # Right knee [2]
        
        left_knee = self.getBodyAngle(results, 23, 25, 27)
        angleList.append(left_knee)
        cv2.putText(image, str(left_knee), self.getBodyPlace(image, results, 25), 1, 2, (0,0,255)) # Left knee [3]

        right_shoulder = self.getBodyAngle(results, 14, 12, 24)
        angleList.append(right_shoulder)
        cv2.putText(image, str(right_knee), self.getBodyPlace(image, results, 12), 1, 2, (0,0,255)) # Right shoulder [4]

        left_shoulder = self.getBodyAngle(results, 13, 11, 23)
        angleList.append(left_shoulder)
        cv2.putText(image, str(left_shoulder), self.getBodyPlace(image, results, 11), 1, 2, (0,0,255)) # Left shoulder [5]

        right_wrist = self.getBodyAngle(results, 14, 16, 20)
        angleList.append(right_wrist)
        cv2.putText(image, str(right_wrist), self.getBodyPlace(image, results, 16), 1, 2, (0,0,255)) # Right wrist [6]

        left_wrist = self.getBodyAngle(results, 13, 15, 19)
        angleList.append(left_wrist)
        cv2.putText(image, str(left_wrist), self.getBodyPlace(image, results, 15), 1, 2, (0,0,255)) # Left wrist [7]

        # Shoulder Slope
        x1 = results.pose_landmarks.landmark[12].x
        y1 = results.pose_landmarks.landmark[12].y
        x2 = results.pose_landmarks.landmark[11].x
        y2 = results.pose_landmarks.landmark[11].y
        slope = (y2-y1)/(x2-x1)
        angleList.append(slope)
        cv2.putText(image, str(slope), (int((x1+x2)/2),int((y1+y2)/2)), 1, 2, (0,0,255)) # Shoulder Slope [8]


        return angleList

    # Purpose is to take a video, analyze it, and save it to file
    # Make sure that the video size is consistant
    def analyzeVideo(self, video_path):
        print("Analyzing video please wait... ")
        vid_capture = cv2.VideoCapture(video_path) # For webcam change <file_path.mp4> to 0. Eg. vid_capture = cv2.VideoCapture(0)
        filename = video_path.split('.')[0] + '_output.mp4'
        video_output = save_file(vid_capture, filename)

        if (vid_capture.isOpened() == False):
            print("Error opening the video file")

        video_angles = []
        count = 0
        while(vid_capture.isOpened()):
            # vid_capture.read() methods returns two values, first element is a boolean
            # and the second is frame
            result, frame = vid_capture.read()
            if result == True:
                # Resize Frame
                frame = resize_img(frame)

                try:
                    # Analyze Video
                    frame, results = self.analyzepic(frame)
                    frame = cv2.flip(frame, 1)
                    frame_angles = self.drawAllAngles(frame, results)
                    video_angles.append(frame_angles)
                except Exception as e:
                    print('Body not found in frame: ' + count)
                    video_angles.append([-1, -1, -1, -1, -1, -1, -1, -1, 0])

                # Save video
                video_output.write(frame)

            else:
                break

            count += 1
        
        video_angles = np.array(video_angles)
        csv_file = video_path.split('.')[0] + '_output.csv'
        np.savetxt(csv_file, video_angles, delimiter=',')

        # Release the video capture object
        vid_capture.release()
        video_output.release()
        print("Analyzed video saved...")

        return filename, csv_file

    def liveWebCam(self, video_path, csv_path):

        print("Opening webcam...")

        # Open video 
        webcam = cv2.VideoCapture(0)
        video = cv2.VideoCapture(video_path)

        # Video settings
        webcam_fps = webcam.get(5)

        out = cv2.VideoWriter("analyzevideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), webcam_fps, (1280, 480))
        Overlay1 = overlay()
        # Open angle file
        video_angles = np.loadtxt(csv_path, delimiter=",")

        if ((webcam.isOpened() == False) and (video.isOpened() == False)):
            print("Error opening one of the video files")

        rowIndex = 0
        right_elbow_correct = 0
        left_elbow_correct = 0
        right_knee_correct = 0
        left_knee_correct = 0
        right_shoulder_correct = 0
        left_shoulder_correct = 0
        right_wrist_correct = 0
        left_wrist_correct = 0
        shoulder_slope_correct = 0
        while(webcam.isOpened() and video.isOpened()):

            # Extract a single frame from each footage
            webcam_success, webcam_frame = webcam.read()
            video_success, video_frame = video.read()

            if webcam_success and video_success:
                try:
                    # Live Analysis of Webcam
                    webcam_frame, landmarks = self.analyzepic(webcam_frame)

                    # Flip webcam frame so that it can reflect the user's movements as a mirror
                    webcam_frame = cv2.flip(webcam_frame, 1)
                    webcam_frame_angles = self.drawAllAngles(webcam_frame, landmarks)
                    video_frame_angles = video_angles[rowIndex]

                    # Add overlay
                    webcam_frame, accuracyList = Overlay1.displayUI(webcam_frame, webcam_frame_angles, video_frame_angles)
                        
                    if accuracyList[0]:
                        right_elbow_correct = right_elbow_correct + 1
                    
                    if accuracyList[1]:
                        left_elbow_correct = left_elbow_correct + 1 

                    if accuracyList[2]:
                        right_knee_correct = right_knee_correct + 1 
                    
                    if accuracyList[3]:
                        left_knee_correct = left_knee_correct + 1 
                    
                    if accuracyList[4]:
                       right_shoulder_correct = right_shoulder_correct + 1
                    
                    if accuracyList[5]:
                       left_shoulder_correct = left_shoulder_correct + 1
                    
                    if accuracyList[6]:
                       right_wrist_correct = right_wrist_correct + 1
                    
                    if accuracyList[7]:
                       left_wrist_correct = left_wrist_correct + 1

                    if accuracyList[8]:
                       shoulder_slope_correct = shoulder_slope_correct + 1

                    
                                           
                except Exception as e:
                    print("Please position your body in a clear view of your webcam...")

                rowIndex = rowIndex + 1

                # Combine the two final frames into one image and display
                image = np.concatenate((webcam_frame, video_frame), axis=1)
                
                out.write(image)

                cv2.imshow('Frame', image)
                # 0 is in milliseconds, try to increase the value, say 50 and observe
                delay = int(1000 / webcam_fps)
                key = cv2.waitKey(delay)

                if key == ord('q'):
                    break
            else:
                break 
        
        # Results Summary
        print()
        print("==============================================")
        print("Accuracy Summary")
        print("==============================================")
        print("Right Elbow Accuracy:  %.2f" % ((right_elbow_correct / rowIndex) * 100) + "%")
        print("Left Elbow Accuracy:  %.2f" % ((left_elbow_correct / rowIndex) * 100) + "%")
        print("Right Knee Accuracy:  %.2f" % ((right_knee_correct / rowIndex) * 100) + "%")
        print("Left Knee Accuracy:  %.2f" % ((left_knee_correct / rowIndex) * 100) + "%")
        print("Right Shoulder Accuracy:  %.2f" % ((right_shoulder_correct / rowIndex) * 100) + "%")
        print("Left Shoulder Accuracy:  %.2f" % ((left_shoulder_correct / rowIndex) * 100) + "%")
        print("Right Wrist Accuracy:  %.2f" % ((right_wrist_correct / rowIndex) * 100) + "%")
        print("Left Wrist Accuracy:  %.2f" % ((left_wrist_correct/ rowIndex) * 100) + "%")
        print("Shoulder Slope Accuracy:  %.2f" % ((shoulder_slope_correct/ rowIndex) * 100) + "%")
        
        webcam.release()
        video.release()
        cv2.destroyAllWindows()

    def mirrorDisplay(self,video_path, csv_path):
        # Open video 
        webcam = cv2.VideoCapture(0)
        video = cv2.VideoCapture(video_path)

        # Video settings
        webcam_fps = webcam.get(5)
        video_fps = video.get(5)
        print("video fps", video_fps)
        print("webcam fps", webcam_fps)


        out = cv2.VideoWriter("analyzevideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), webcam_fps, (1280, 480))
        Overlay1 = overlay()
        # Open angle file
        video_angles = np.loadtxt(csv_path, delimiter=",")

        # Get mirror background
        blackbg = cv2.imread('samples/resources/blackbg.png')
        print(blackbg.shape)
        blackbg = resize_img(blackbg)
        
        if ((webcam.isOpened() == False) and (video.isOpened() == False)):
            print("Error opening one of the video files")

        rowIndex = 0
        right_elbow_correct = 0
        while(webcam.isOpened() and video.isOpened()):

            # Extract a single frame from each footage
            webcam_success, webcam_frame = webcam.read()
            video_success, video_frame = video.read()

            if webcam_success and video_success:
                try:
                    # Live Analysis of Webcam
                    webcam_frame, landmarks = self.analyzepic(webcam_frame)

                    # Flip webcam frame so that it can reflect the user's movements as a mirror
                    webcam_frame = cv2.flip(webcam_frame, 1)
                    webcam_frame_angles = self.drawAllAngles(webcam_frame, landmarks)
                    video_frame_angles = video_angles[rowIndex]

                    # Add overlay
                    blackbg, right_arm_value = Overlay1.displayUI(blackbg, webcam_frame_angles, video_frame_angles)
                        
                    if right_arm_value:
                        right_elbow_correct = right_elbow_correct + 1

                except Exception as e:
                    print(e)
                    print("error could not read webcam")

                rowIndex = rowIndex + 1

                video_frame = cv2.resize(video_frame, (800,960))

                # Combine the two final frames into one image and display
                image = np.concatenate((blackbg, video_frame), axis=1)
                
                out.write(image)

                cv2.imshow('Frame', image)
                # 0 is in milliseconds, try to increase the value, say 50 and observe
                delay = int(1000 / webcam_fps)
                key = cv2.waitKey(delay)

                if key == ord('q'):
                    break
            else:
                break 

        print("Your right elbow was accurate for " + str(right_elbow_correct / rowIndex) + "%")

        webcam.release()
        video.release()
        cv2.destroyAllWindows()