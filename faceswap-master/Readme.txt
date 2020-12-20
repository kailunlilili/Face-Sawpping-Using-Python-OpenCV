CIS 581 Final Project Documentation

Requirements: 
Python 3, numpy, OpenCV, dlib, os, ffmpeg


Instructions:
Download, unzip, and move to main directory shape_predictor_68_face_landmarks.dat.bz2 if there isn't one in the main directory.
Download from: https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/download


Main Task: TwoVideosFaceSwap.py
1. Run TwoVideosFaceSwap.py, the test result will store in results folder. The result image for each frame will store in the output folder.
2. To change test videos, change 'source_path', 'target_path' variable to desired video path.
3. We will swap the face in the source video onto the body of target video. 


Sub Task: RealTimeFaceSwap.py
1. Run RealTimeFaceSwap.py in terminal using command python3 RealTimeFaceSwap.py.
2. A window will pop up for face swapping, please make sure there is a single face in the frame.
3. Press Q to close the window.
4. To change source image of face, change 'source_image_path' to the desired image path. 
5. Wearing glasses will worsen the swapping result. I would suggest you to try it without glasses. 


Sub Task: SingVideoTwoFacesSwap.py
1. Run SingVideoTwoFacesSwap.py the test result will store in single_video_results folder. The result image for each frame will store in the single_video_output folder.
2. To change test videos, change 'target_video_path' to desired video path. Please make sure there are two faces in the video. 



