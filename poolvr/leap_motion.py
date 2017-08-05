import ctypes

import numpy as np
import cv2

import Leap


controller = Leap.Controller()
controller.set_policy(Leap.Controller.POLICY_BACKGROUND_FRAMES)
controller.set_policy(Leap.Controller.POLICY_IMAGES)

fast = cv2.FastFeatureDetector_create()

def main():
    while True:
        image0 = controller.images[0]
        image1 = controller.images[1]
        if not image0.is_valid or not image1.is_valid:
            continue
        L = np.ctypeslib.as_array((ctypes.c_ubyte * image0.width * image0.height).from_address(int(image0.data_pointer)))
        R = np.ctypeslib.as_array((ctypes.c_ubyte * image1.width * image1.height).from_address(int(image1.data_pointer)))
        #cv2.imshow('left', L)
        kp = fast.detect(L, None)
        cv2.imshow('left', cv2.drawKeypoints(L, kp, None, color=(255,255,0)))
        edges = cv2.Canny(L, 140, 200)
        cv2.imshow('edges', edges)
        cv2.imshow('right', R)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
   main()
