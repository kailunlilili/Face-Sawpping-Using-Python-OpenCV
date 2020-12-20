import cv2
import dlib
import numpy as np


def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float64)


def ref2dImagePoints(shape):
    imagePoints = [[shape.part(30).x, shape.part(30).y],
                   [shape.part(8).x, shape.part(8).y],
                   [shape.part(36).x, shape.part(36).y],
                   [shape.part(45).x, shape.part(45).y],
                   [shape.part(48).x, shape.part(48).y],
                   [shape.part(54).x, shape.part(54).y]]
    return np.array(imagePoints, dtype=np.float64)


def CameraMatrix(fl, center):
    cameraMatrix = [[fl, 1, center[0]],
                    [0, fl, center[1]],
                    [0, 0, 1]]
    return np.array(cameraMatrix, dtype=np.float)


def pose(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    face3Dmodel = ref3DModel()
    for face in faces:
        # detect facial landmarks
        shape = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face)
        refImgPts = ref2dImagePoints(shape)
        height, width, channels = img.shape
        focalLength = 1 * width
        cameraMatrix = CameraMatrix(focalLength, (height / 2, width / 2))
        mdists = np.zeros((4, 1), dtype=np.float64)

        # calculate rotation and translation vector using solvePnP
        success, rotationVector, translationVector = cv2.solvePnP(
            face3Dmodel, refImgPts, cameraMatrix, mdists)

        rmat, jac = cv2.Rodrigues(rotationVector)
        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

        gaze = 'Forward'
        if angles[1] < -30:
            gaze = "Left"
        elif angles[1] > 30:
            gaze = "Right"
        else:
            gaze = "Forward"
        return gaze
