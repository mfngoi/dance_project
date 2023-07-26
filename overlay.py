import cv2
import numpy as np


class overlay:

    def __init__(self):
        self.overlay_image = cv2.imread('samples/resources/human.png')
        self.overlay_image = cv2.resize(self.overlay_image, (self.overlay_image.shape[1]//4,self.overlay_image.shape[0]//4))

    def displayUI(self, frame, angle1, angle2):
        _image = self.overlay_image.copy()

        accuracyList = []
        
        difference = np.abs(angle1[0] - angle2[0])
        if difference > 40:  
            cv2.circle(_image, (137,112), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)

        
        difference = np.abs(angle1[1] - angle2[1])
        if difference > 40:  
            cv2.circle(_image, (56,112), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)


        difference = np.abs(angle1[2] - angle2[2])
        if difference > 40:  
            cv2.circle(_image, (115,198), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)

        
        difference = np.abs(angle1[3] - angle2[3])
        if difference > 40: 
            cv2.circle(_image, (76,198), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)

        difference = np.abs(angle1[4] - angle2[4]) # right_shoulder
        if difference > 40: 
            cv2.circle(_image, (70,55), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)
        
        difference = np.abs(angle1[5] - angle2[5]) #  left_shoulder 
        if difference > 40: 
            cv2.circle(_image, (130,55), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)
        
        difference = np.abs(angle1[6] - angle2[6]) # right_wrist
        if difference > 40: 
            cv2.circle(_image, (57,125), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)
                
        difference = np.abs(angle1[7] - angle2[7]) #  left_wrist 
        if difference > 40: 
            cv2.circle(_image, (145,125), 10, (0,0,255), -1)
            accuracyList.append(False)
        else:
            accuracyList.append(True)
            
        
        # Shoulder Slope
        slope1 = angle1[8]
        slope2 = angle2[8]
        if (slope1 > 0 and slope2 > 0) or (slope1 < 0 and slope2 < 0):
            if np.abs(slope1-slope2) > 0.7: 
                cv2.line(_image, (70,55), (130,55), (0,0,255), 3)
                accuracyList.append(False)
            else:
                accuracyList.append(True)
        else:
            cv2.line(_image, (70,55), (130,55), (0,0,255), 3)
            accuracyList.append(False)
        

        height = _image.shape[0]
        width = _image.shape[1]

        # # replace values at coordinates (100, 100) to (399, 399) of img3 with region of img2
        frame[0:height, 0:width,:] = _image[0:height, 0:width,:]

        return frame, accuracyList