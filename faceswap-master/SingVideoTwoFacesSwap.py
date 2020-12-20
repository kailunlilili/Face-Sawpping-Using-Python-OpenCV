import cv2
import dlib
import numpy as np

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

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
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
    img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
# -----------------------------修改部分-------------------------------#

def face_swap_in_a_single_video(video_path):
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
    # video reader and writer
    cap = cv2.VideoCapture(rawVideo)

    trackVideo = 'single_video_results/Output_' + rawVideo[-6:]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)) )
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    frame_cnt = 0

    # read first frame to capture two faces
    # and save the triangulation of the two faces
    ret, frame = cap.read()
    im = frame.copy()
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    rects = detector(im, 1)
    if rects[0].left() < rects[1].left():
        im1_origin, landmarks1 = (im, predictor(im, rects[0]))
        im2_origin, landmarks2 = (im, predictor(im, rects[1]))
    else:
        im1_origin, landmarks1 = (im, predictor(im, rects[1]))
        im2_origin, landmarks2 = (im, predictor(im, rects[0]))


    landmarks = landmarks1
    landmarks_points_origin_1 = []
    for n in ALIGN_POINTS:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points_origin_1.append((x, y))

    points = np.array(landmarks_points_origin_1, np.int32)
    convexhull1_origin = cv2.convexHull(points)
    triangles = delaunay_triangulation(convexhull1_origin, landmarks_points_origin_1)
    indexes_triangles1 = []
    get_indexes_triangles(indexes_triangles1, triangles, points)
    # -----------------------------
    landmarks = landmarks2
    landmarks_points_origin_2 = []
    for n in ALIGN_POINTS:
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points_origin_2.append((x, y))

    points = np.array(landmarks_points_origin_2, np.int32)
    convexhull2_origin = cv2.convexHull(points)
    triangles = delaunay_triangulation(convexhull2_origin, landmarks_points_origin_2)
    indexes_triangles2 = []
    get_indexes_triangles(indexes_triangles2, triangles, points)

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if np.shape(frame) == ():
            continue

        frame_cnt += 1


        im = frame.copy()
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,im.shape[0] * SCALE_FACTOR))
        rects = detector(im, 1)
        if len(rects) != 2:
            print("less than or more than two faces. skipping.")
            continue

        if rects[0].left() < rects[1].left():
            im1, landmarks1 = (im, predictor(im, rects[0]))
            im2, landmarks2 = (im, predictor(im, rects[1]))
        else:
            im1, landmarks1 = (im, predictor(im, rects[1]))
            im2, landmarks2 = (im, predictor(im, rects[0]))

        landmarks_points1 = []
        for n in ALIGN_POINTS:
            x = landmarks1.part(n).x
            y = landmarks1.part(n).y
            landmarks_points1.append((x, y))

        landmarks_points2 = []
        for n in ALIGN_POINTS:
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            landmarks_points2.append((x, y))

        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height, width, channels = im.shape

        img2_new_face = np.zeros((height, width, channels), np.uint8)

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        points1 = np.array(landmarks_points1, np.int32)
        convexhull1 = cv2.convexHull(points1)

        lines_space_mask = np.zeros_like(img_gray)

        for triangle_index in indexes_triangles1:
            recon_face(landmarks_points_origin_1, landmarks_points2, triangle_index, img2_new_face, im1_origin, lines_space_mask)

        img2_face_mask = np.zeros_like(img_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(im, im, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone_mouth_mix = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.MIXED_CLONE)

        # # create a mouth mask
        mouth_points2 = []
        for n in MOUTH_POINTS:
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            mouth_points2.append((x, y))

        mouthpoints2 = np.array(mouth_points2, np.int32)
        mouthconvexhull2 = cv2.convexHull(mouthpoints2)
        img2_mouth_mask = np.zeros_like(img_gray)
        img2_mouth_mask = cv2.fillConvexPoly(img2_mouth_mask, mouthconvexhull2, 255)
        img2_mouth_mask = cv2.cvtColor(img2_mouth_mask, cv2.COLOR_GRAY2BGR)
        # extrac mixed mouth part
        img2_mouth_part = seamlessclone_mouth_mix & img2_mouth_mask
        # seamlessclone
        seamlessclone = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        # replace mouth part on the final image with mixed mouth
        seamlessclone = cv2.subtract(seamlessclone, img2_mouth_mask)
        seamlessclone = cv2.add(seamlessclone, img2_mouth_part)

        img1_new_face = np.zeros((height, width, channels), np.uint8)
        for triangle_index in indexes_triangles2:
            recon_face(landmarks_points_origin_2, landmarks_points1, triangle_index, img1_new_face, im2_origin, lines_space_mask)

        img1_face_mask = np.zeros_like(img_gray)
        img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull1, 255)
        img1_face_mask = cv2.bitwise_not(img1_head_mask)
        img2_head_noface = cv2.bitwise_and(im, im, mask=img1_face_mask)

        result1 = cv2.add(img2_head_noface, img1_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull1)
        center_face1 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone_mouth_mix = cv2.seamlessClone(result1, im, img1_head_mask, center_face1, cv2.MIXED_CLONE)

        # # create a mouth mask
        mouth_points2 = []
        for n in MOUTH_POINTS:
            x = landmarks1.part(n).x
            y = landmarks1.part(n).y
            mouth_points2.append((x, y))

        mouthpoints2 = np.array(mouth_points2, np.int32)
        mouthconvexhull2 = cv2.convexHull(mouthpoints2)
        img2_mouth_mask = np.zeros_like(img_gray)
        img2_mouth_mask = cv2.fillConvexPoly(img2_mouth_mask, mouthconvexhull2, 255)
        img2_mouth_mask = cv2.cvtColor(img2_mouth_mask, cv2.COLOR_GRAY2BGR)
        # extrac mixed mouth part
        img2_mouth_part = seamlessclone_mouth_mix & img2_mouth_mask
        # seamlessclone
        seamlessclone1 = cv2.seamlessClone(result1, im, img1_head_mask, center_face1, cv2.NORMAL_CLONE)

        # replace mouth part on the final image with mixed mouth
        seamlessclone1 = cv2.subtract(seamlessclone1, img2_mouth_mask)
        seamlessclone1 = cv2.add(seamlessclone1, img2_mouth_part)


        img2_head_mask = cv2.cvtColor(img2_head_mask, cv2.COLOR_GRAY2BGR)

        right_head = seamlessclone & img2_head_mask
        left_head_no_right_head = cv2.subtract(seamlessclone1, img2_head_mask)
        output_im = cv2.add(left_head_no_right_head, right_head)

        writer.write(output_im)
        cv2.imwrite('single_video_output/{}.jpg'.format(frame_cnt), output_im)
        print("finished adding frame -- ", frame_cnt)
    print('saving now -----------')
    cv2.destroyAllWindows()
    cap.release()
    writer.release()

if __name__ == "__main__":
    target_video_path = './test video/SingleVideoTest1.mp4'
    face_swap_in_a_single_video(target_video_path)
    print("face swap finished")