# video to image  ->>>>>>>>  ffmpeg -i testvideo1.mov output%05d.jpg
# image to video ->>>>>>>> ffmpeg -i output%05d.jpg out.mp4


import cv2
import dlib
import numpy
from face_pose_detection_image import *
from Video_Pose_Detection import *
class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass

def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index



# -----------------------------修改部分-------------------------------#
def correct_colours(im1, im2, blur_amount):
    blur_amount = int(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    im2_blur += 1

    return ((im2.astype(numpy.float64) / im2_blur.astype(numpy.float64)) * im1_blur.astype(numpy.float64)).astype(
        np.uint8)


def delaunay_triangulation(convexhull,landmarks_points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    return triangles


def get_indexes_triangles(indexes_triangles, triangles, points):
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)


def line_space(tr1_pt1, tr1_pt2, tr1_pt3, lines_space_mask, im_source):
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
    cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
    cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
    lines_space = cv2.bitwise_and(im_source, im_source, mask=lines_space_mask)
    return lines_space


def triangulation_face1(landmarks_points,triangle_index,im_source ):
    tr1_pt1 = landmarks_points[triangle_index[0]]
    tr1_pt2 = landmarks_points[triangle_index[1]]
    tr1_pt3 = landmarks_points[triangle_index[2]]
    triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

    rect1 = cv2.boundingRect(triangle1)
    (x, y, w, h) = rect1
    cropped_triangle = im_source[y: y + h, x: x + w]
    cropped_tr1_mask = np.zeros((h, w), np.uint8)

    points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                       [tr1_pt2[0] - x, tr1_pt2[1] - y],
                       [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
    return tr1_pt1, tr1_pt2, tr1_pt3, cropped_triangle, cropped_tr1_mask, points


def triangulation_face2(landmarks_points2,triangle_index,im_source ):
    tr2_pt1 = landmarks_points2[triangle_index[0]]
    tr2_pt2 = landmarks_points2[triangle_index[1]]
    tr2_pt3 = landmarks_points2[triangle_index[2]]
    triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

    rect2 = cv2.boundingRect(triangle2)
    (x, y, w, h) = rect2

    cropped_tr2_mask = np.zeros((h, w), np.uint8)

    points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                        [tr2_pt2[0] - x, tr2_pt2[1] - y],
                        [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

    cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
    return tr2_pt2, tr2_pt2, tr2_pt3, cropped_tr2_mask, points2, rect2


def recon_face (landmarks_points, landmarks_points2, triangle_index, img2_new_face, im_source, lines_space_mask):
    # Triangulation of the first face
    tr1_pt1, tr1_pt2, tr1_pt3, cropped_triangle, cropped_tr1_mask, points= triangulation_face1(landmarks_points,triangle_index,im_source)

    # Lines space
    lines_space = line_space(tr1_pt1, tr1_pt2, tr1_pt3, lines_space_mask, im_source)

    # Triangulation of second face
    tr2_pt2, tr2_pt2, tr2_pt3, cropped_tr2_mask, points2, rect2 = triangulation_face2(landmarks_points2, triangle_index, im_source)
    (x, y, w, h) = rect2

    # Warp triangles
    points = np.float32(points)
    points2 = np.float32(points2)
    M = cv2.getAffineTransform(points, points2)
    warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

    # Reconstructing destination face
    img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    # if mask_triangles_designed.shape != warped_triangle[:3]:
    #     return 'error'
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    # return 'no error'
# -----------------------------修改部分-------------------------------#


def face_swap_between_image_and_video(video_path):
    rawVideo = video_path
    PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
    SCALE_FACTOR = 1
    FACE_POINTS = list(range(17, 68))
    MOUTH_POINTS = list(range(48, 68))
    # Teeth_POINTS = list(range(61, 68))
    RIGHT_BROW_POINTS = list(range(17, 22))
    LEFT_BROW_POINTS = list(range(22, 27))
    RIGHT_EYE_POINTS = list(range(36, 42))
    LEFT_EYE_POINTS = list(range(42, 48))
    NOSE_POINTS = list(range(27, 35))
    JAW_POINTS = list(range(0, 17))

    # Points used to line up the images.
    # ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
    #                 RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
    ALIGN_POINTS = list(range(68))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)


# image gray
    im_source_dict = dict()
    img_gray_dict = dict()
    indexes_triangles_dict = dict()
    landmarks_points_dict = dict()

    pose_dict = {'Left':'./GAZE/left.jpg','Forward': './GAZE/forward.jpg','Right': './GAZE/right.jpg'}
    for pose_name, im_pth in pose_dict.items():
        sourceImage = im_pth
        im_source = cv2.imread(sourceImage)
        im_source_dict[pose_name] = im_source
        im_source = cv2.resize(im_source, (im_source.shape[1] * SCALE_FACTOR,
                                           im_source.shape[0] * SCALE_FACTOR))

        img_gray = cv2.cvtColor(im_source, cv2.COLOR_BGR2GRAY)
        img_gray_dict[pose_name] = img_gray
        source_rect = detector(im_source, 1)
        faces = detector(img_gray)


        for face in faces:
            landmarks = predictor(img_gray, face)
            landmarks_points = []
            for n in ALIGN_POINTS:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
            landmarks_points_dict[pose_name] = landmarks_points
            points = np.array(landmarks_points, np.int32)
            convexhull = cv2.convexHull(points)
            triangles = delaunay_triangulation(convexhull,landmarks_points)
            indexes_triangles = []
            get_indexes_triangles(indexes_triangles, triangles,points)
            indexes_triangles_dict[pose_name] = indexes_triangles
    # video reader and writer
    cap = cv2.VideoCapture(rawVideo)

    #check if need rotate
    # rotateCode = check_rotation(rawVideo)

    # Initialize video writer for tracking video
    trackVideo = 'results/Output_' + rawVideo[-6:]
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)) )
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    frame_cnt = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            print('no ret break')
            break

        if np.shape(frame) == ():
            print('not frame continue')
            continue
        # if rotateCode is not None:
        #     frame = correct_rotation(frame, rotateCode)

    # rotate if iphone videos, comment if download videos
    #     frame = correct_rotation(frame, cv2.ROTATE_180)
    # ------------------------------------------------

        frame_cnt += 1

        # im = cv2.imread(filename)
        im = frame.copy()
        frame_pose = pose(im)
        if not frame_pose:
            print(frame_cnt, '  No Face Pose')
            continue
        print('frame --- ',frame_cnt, 'pose is ', frame_pose)

        im_source = im_source_dict[frame_pose]
        img_gray = img_gray_dict[frame_pose]
        indexes_triangles = indexes_triangles_dict[frame_pose]
        landmarks_points = landmarks_points_dict[frame_pose]


        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                             im.shape[0] * SCALE_FACTOR))
        img2_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height, width, channels = im.shape
        img2_new_face = np.zeros((height, width, channels), np.uint8)
        rects = detector(img2_gray)

        if len(rects) == 0:
            print("missing faces. skipping.")
            continue

        if len(rects) > 1:
            print("more than one face. skipping.")
            continue

        landmarks = predictor(img2_gray, rects[0])
        landmarks_points2 = []
        for n in ALIGN_POINTS:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        lines_space_mask = np.zeros_like(img_gray)
        lines_space_new_face = np.zeros_like(im)
        try:

            for triangle_index in indexes_triangles:
                recon_face(landmarks_points, landmarks_points2, triangle_index, img2_new_face, im_source,lines_space_mask)
        except:
            print('error here, jump to next frame')
            continue

        # Face swapped (putting 1st face into 2nd face)
        img2_face_mask = np.zeros_like(img2_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        # -----------------------------Color Correction-------------------------------#
        img_face_mask = cv2.bitwise_not(img2_face_mask)

        f = cv2.bitwise_and(im, im, mask=img_face_mask)
        img2_new_face = correct_colours(img2_new_face, f, 3)

        # #f 和 newface
        f1_hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

        f1h = f1_hsv[:, :, 0]
        f1h_mean = f1h[np.nonzero(f1h)].mean()

        f1s = f1_hsv[:, :, 1]
        f1s_mean = f1s[np.nonzero(f1s)].mean()

        f1v = f1_hsv[:, :, 2]
        f1v_mean = f1v[np.nonzero(f1v)].mean()

        f2_hsv = cv2.cvtColor(img2_new_face, cv2.COLOR_BGR2HSV)
        f2h = f2_hsv[:, :, 0]
        f2h_mean = f2h[np.nonzero(f2h)].mean()

        f2s = f2_hsv[:, :, 1]
        f2s_mean = f2s[np.nonzero(f2s)].mean()

        f2v = f2_hsv[:, :, 2]
        f2v_mean = f2v[np.nonzero(f2v)].mean()

        hs = ((f1h_mean - f2h_mean)).astype(np.uint8)
        # print(type(hs))
        ss = ((f1s_mean - f2s_mean) / 2).astype(np.uint8)
        vs = ((f1v_mean - f2v_mean) / 2).astype(np.uint8)

        # f2h[np.nonzero(f2h)] += hs
        # f2h[f2h > 179] -= 179
        # f2h[f2h < 0] += 179

        f2s[np.logical_or(f2h > 130, f2h < 100)] = f2s_mean
        f2s = np.clip(f2s, 0, 255)

        # f2v[np.nonzero(f2v)] += vs
        # print('vs=',vs)
        # f2v = np.clip(f2v, 0, 255)

        f2_hsv[:, :, 0] = f2h
        f2_hsv[:, :, 1] = f2s
        f2_hsv[:, :, 2] = f2v
        img2_new_face = cv2.cvtColor(f2_hsv, cv2.COLOR_HSV2BGR)
        # -----------------------------Color Correction-------------------------------#

        img2_head_noface = cv2.bitwise_and(im, im, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        # create a mixed clone for mouth part
        seamlessclone_mouth_mix = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.MIXED_CLONE)

        # # create a mouth mask
        mouth_points2 = []
        for n in MOUTH_POINTS:
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            mouth_points2.append((x, y))

        mouthpoints2 = np.array(mouth_points2, np.int32)
        mouthconvexhull2 = cv2.convexHull(mouthpoints2)
        img2_mouth_mask = np.zeros_like(img2_gray)
        img2_mouth_mask = cv2.fillConvexPoly(img2_mouth_mask, mouthconvexhull2, 255)
        img2_mouth_mask = cv2.cvtColor(img2_mouth_mask, cv2.COLOR_GRAY2BGR)
        # Use mouth part from current frame
        # img2_mouth_part = im & img2_mouth_mask
        # Use mouth part from mix of current frame and source image
        img2_mouth_part = seamlessclone_mouth_mix & img2_mouth_mask

        result_no_mouth = cv2.subtract(result, img2_mouth_mask)
        result_plus_img1_mouth = cv2.add(result_no_mouth, img2_mouth_part)

        # -----------------------------Color Correction-------------------------------#
        mouth_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        mh = mouth_hsv[:, :, 0]
        ms = mouth_hsv[:, :, 1]
        mv = mouth_hsv[:, :, 2]
        ms[np.logical_or(mh > 130, mh < 100)] = f2s_mean
        ms = np.clip(ms, 0, 255)

        mouth_hsv[:, :, 0] = mh
        mouth_hsv[:, :, 1] = ms
        mouth_hsv[:, :, 2] = mv
        result = cv2.cvtColor(mouth_hsv, cv2.COLOR_HSV2BGR)
        # -----------------------------Color Correction-------------------------------#

        # seamlessclone = cv2.seamlessClone(result_plus_img1_mouth, im, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        seamlessclone = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
        final_no_mouth = cv2.subtract(seamlessclone, img2_mouth_mask)
        final = cv2.add(final_no_mouth, img2_mouth_part)
        writer.write(final)
        cv2.imwrite('output/{}.jpg'.format(frame_cnt), final)
        print("finished adding frame -- ", frame_cnt)
    print('saving now -----------')
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video1_path = './test video/LucianoRosso1.mp4'
    video2_path = './test video/dance2.mp4'
    video_pose_detection(video1_path)
    face_swap_between_image_and_video(video2_path)
    video_pose_detection(video2_path)
    face_swap_between_image_and_video(video1_path)
    print("face swap finished")