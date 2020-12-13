# import cv2
# import dlib
# import numpy
# import numpy as np
# import glob
#
# PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
# SCALE_FACTOR = 1
# FEATHER_AMOUNT = 11
#
# FACE_POINTS = list(range(17, 68))
# MOUTH_POINTS = list(range(48, 61))
# RIGHT_BROW_POINTS = list(range(17, 22))
# LEFT_BROW_POINTS = list(range(22, 27))
# RIGHT_EYE_POINTS = list(range(36, 42))
# LEFT_EYE_POINTS = list(range(42, 48))
# NOSE_POINTS = list(range(27, 35))
# JAW_POINTS = list(range(0, 17))
#
# # Points used to line up the images.
# ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
#                 RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
#
# # Points from the second image to overlay on the first. The convex hull of each
# # element will be overlaid.
# OVERLAY_POINTS = [
#     LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS,
#     NOSE_POINTS + MOUTH_POINTS,
# ]
#
# # Amount of blur to use during colour correction, as a fraction of the
# # pupillary distance.
# COLOUR_CORRECT_BLUR_FRAC = 0.6
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(PREDICTOR_PATH)
#
#
# class TooManyFaces(Exception):
#     pass
#
#
# class NoFaces(Exception):
#     pass
#
#
# def annotate_landmarks(im, landmarks):
#     im = im.copy()
#     for idx, point in enumerate(landmarks):
#         pos = (point[0, 0], point[0, 1])
#         cv2.putText(im, str(idx), pos,
#                     fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
#                     fontScale=0.4,
#                     color=(0, 0, 255))
#         cv2.circle(im, pos, 3, color=(0, 255, 255))
#     return im
#
#
# def draw_convex_hull(im, points, color):
#     points = cv2.convexHull(points)
#     cv2.fillConvexPoly(im, points, color=color)
#
#
# def get_face_mask(im, landmarks):
#     im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
#
#     for group in OVERLAY_POINTS:
#         draw_convex_hull(im,
#                          landmarks[group],
#                          color=1)
#
#     im = numpy.array([im, im, im]).transpose((1, 2, 0))
#
#     im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
#     im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
#
#     return im
#
#
# def transformation_from_points(points1, points2):
#     """
#     Return an affine transformation [s * R | T] such that:
#
#         sum ||s*R*p1,i + T - p2,i||^2
#
#     is minimized.
#
#     """
#     # Solve the procrustes problem by subtracting centroids, scaling by the
#     # standard deviation, and then using the SVD to calculate the rotation. See
#     # the following for more details:
#     #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
#
#     points1 = points1.astype(numpy.float64)
#     points2 = points2.astype(numpy.float64)
#
#     c1 = numpy.mean(points1, axis=0)
#     c2 = numpy.mean(points2, axis=0)
#     points1 -= c1
#     points2 -= c2
#
#     s1 = numpy.std(points1)
#     s2 = numpy.std(points2)
#     points1 /= s1
#     points2 /= s2
#
#     U, S, Vt = numpy.linalg.svd(points1.T * points2)
#
#     # The R we seek is in fact the transpose of the one given by U * Vt. This
#     # is because the above formulation assumes the matrix goes on the right
#     # (with row vectors) where as our solution requires the matrix to be on the
#     # left (with column vectors).
#     R = (U * Vt).T
#
#     return numpy.vstack([numpy.hstack(((s2 / s1) * R,
#                                        c2.T - (s2 / s1) * R * c1.T)),
#                          numpy.matrix([0., 0., 1.])])
#
#
# def warp_im(im, M, dshape):
#     output_im = numpy.zeros(dshape, dtype=im.dtype)
#     cv2.warpAffine(im,
#                    M[:2],
#                    (dshape[1], dshape[0]),
#                    dst=output_im,
#                    borderMode=cv2.BORDER_TRANSPARENT,
#                    flags=cv2.WARP_INVERSE_MAP)
#     return output_im
#
#
# def correct_colours(im1, im2, landmarks1):
#     blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
#         numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
#         numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
#     blur_amount = int(blur_amount)
#     if blur_amount % 2 == 0:
#         blur_amount += 1
#     im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
#     im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
#
#     # Avoid divide-by-zero errors.
#     # im2_blur += 128 * (im2_blur <= 1.0)
#     im2_blur = numpy.add(im2_blur, 128 * (im2_blur <= 1.0), out=im2_blur, casting="unsafe")
#     return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
#             im2_blur.astype(numpy.float64))
#
# def extract_index_nparray(nparray):
#     index = None
#     for num in nparray[0]:
#         index = num
#         break
#     return index
#
# # -----------------------
# def get_index_triangle(landmarks):
#     landmarks_points = landmarks.tolist()
#     points = np.array(landmarks_points, np.int32)
#     convexhull = cv2.convexHull(points)
#     # Delaunay triangulation
#     rect = cv2.boundingRect(convexhull)
#     subdiv = cv2.Subdiv2D(rect)
#     subdiv.insert(landmarks_points)
#     triangles = subdiv.getTriangleList()
#     triangles = np.array(triangles, dtype=np.int32)
#     indexes_triangles = []
#     for t in triangles:
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#
#         index_pt1 = np.where((points == pt1).all(axis=1))
#         index_pt1 = extract_index_nparray(index_pt1)
#
#         index_pt2 = np.where((points == pt2).all(axis=1))
#         index_pt2 = extract_index_nparray(index_pt2)
#
#         index_pt3 = np.where((points == pt3).all(axis=1))
#         index_pt3 = extract_index_nparray(index_pt3)
#
#         if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
#             triangle = [index_pt1, index_pt2, index_pt3]
#             indexes_triangles.append(triangle)
#
#     return indexes_triangles, convexhull
#
# def warp_from_source_to_target(landmarks_points, img_gray, indexes_triangles, im, im_source):
#     height, width, channels = im.shape
#     img2_new_face = np.zeros((height, width, channels), np.uint8)
#
#     landmarks_points2 = landmarks_points.tolist()
#     points2 = np.array(landmarks_points2, np.int32)
#     convexhull2 = cv2.convexHull(points2)
#
#     lines_space_mask = np.zeros_like(img_gray)
#     lines_space_new_face = np.zeros_like(im)
#
#     for triangle_index in indexes_triangles:
#         tr1_pt1 = landmarks_points[triangle_index[0]]
#         tr1_pt2 = landmarks_points[triangle_index[1]]
#         tr1_pt3 = landmarks_points[triangle_index[2]]
#         triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
#
#         rect1 = cv2.boundingRect(triangle1)
#         (x, y, w, h) = rect1
#         cropped_triangle = im_source[y: y + h, x: x + w]
#         cropped_tr1_mask = np.zeros((h, w), np.uint8)
#
#         points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
#                            [tr1_pt2[0] - x, tr1_pt2[1] - y],
#                            [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
#
#         cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
#
#         # Lines space
#         cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
#         cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
#         cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
#         lines_space = cv2.bitwise_and(im_source, im_source, mask=lines_space_mask)
#
#         # Triangulation of second face
#         tr2_pt1 = landmarks_points2[triangle_index[0]]
#         tr2_pt2 = landmarks_points2[triangle_index[1]]
#         tr2_pt3 = landmarks_points2[triangle_index[2]]
#         triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
#
#         rect2 = cv2.boundingRect(triangle2)
#         (x, y, w, h) = rect2
#
#         cropped_tr2_mask = np.zeros((h, w), np.uint8)
#
#         points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
#                             [tr2_pt2[0] - x, tr2_pt2[1] - y],
#                             [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
#
#         cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
#
#         # Warp triangles
#         points = np.float32(points)
#         points2 = np.float32(points2)
#         M = cv2.getAffineTransform(points, points2)
#         warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
#         warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
#
#         # Reconstructing destination face
#         img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
#         img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
#         _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
#         warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
#
#         img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
#         img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
#     return img2_new_face, convexhull2
#
#
# # -----------------------
#
# def delaunay_triangulation(convexhull,landmarks_points):
#     rect = cv2.boundingRect(convexhull)
#     subdiv = cv2.Subdiv2D(rect)
#     subdiv.insert(landmarks_points)
#     triangles = subdiv.getTriangleList()
#     triangles = np.array(triangles, dtype=np.int32)
#     return triangles
#
#
# def get_indexes_triangles(indexes_triangles, triangles, points):
#     for t in triangles:
#         pt1 = (t[0], t[1])
#         pt2 = (t[2], t[3])
#         pt3 = (t[4], t[5])
#
#         index_pt1 = np.where((points == pt1).all(axis=1))
#         index_pt1 = extract_index_nparray(index_pt1)
#
#         index_pt2 = np.where((points == pt2).all(axis=1))
#         index_pt2 = extract_index_nparray(index_pt2)
#
#         index_pt3 = np.where((points == pt3).all(axis=1))
#         index_pt3 = extract_index_nparray(index_pt3)
#
#         if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
#             triangle = [index_pt1, index_pt2, index_pt3]
#             indexes_triangles.append(triangle)
#
#
# def line_space(tr1_pt1, tr1_pt2, tr1_pt3, lines_space_mask, im_source):
#     cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
#     cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
#     cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
#     lines_space = cv2.bitwise_and(im_source, im_source, mask=lines_space_mask)
#     return lines_space
#
#
# def triangulation_face1(landmarks_points,triangle_index,im_source ):
#     tr1_pt1 = landmarks_points[triangle_index[0]]
#     tr1_pt2 = landmarks_points[triangle_index[1]]
#     tr1_pt3 = landmarks_points[triangle_index[2]]
#     triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
#
#     rect1 = cv2.boundingRect(triangle1)
#     (x, y, w, h) = rect1
#     cropped_triangle = im_source[y: y + h, x: x + w]
#     cropped_tr1_mask = np.zeros((h, w), np.uint8)
#
#     points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
#                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
#                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)
#
#     cv2.fillConvexPoly(cropped_tr1_mask, points, 255)
#     return tr1_pt1, tr1_pt2, tr1_pt3, cropped_triangle, cropped_tr1_mask, points
#
#
# def triangulation_face2(landmarks_points2,triangle_index,im_source ):
#     tr2_pt1 = landmarks_points2[triangle_index[0]]
#     tr2_pt2 = landmarks_points2[triangle_index[1]]
#     tr2_pt3 = landmarks_points2[triangle_index[2]]
#     triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
#
#     rect2 = cv2.boundingRect(triangle2)
#     (x, y, w, h) = rect2
#
#     cropped_tr2_mask = np.zeros((h, w), np.uint8)
#
#     points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
#                         [tr2_pt2[0] - x, tr2_pt2[1] - y],
#                         [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)
#
#     cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
#     return tr2_pt2, tr2_pt2, tr2_pt3, cropped_tr2_mask, points2, rect2
#
#
# def recon_face (landmarks_points, landmarks_points2, triangle_index, img2_new_face, im_source, lines_space_mask):
#     # Triangulation of the first face
#     tr1_pt1, tr1_pt2, tr1_pt3, cropped_triangle, cropped_tr1_mask, points= triangulation_face1(landmarks_points,triangle_index,im_source)
#
#     # Lines space
#     # lines_space = line_space(tr1_pt1, tr1_pt2, tr1_pt3, lines_space_mask, im_source)
#
#     # Triangulation of second face
#     tr2_pt2, tr2_pt2, tr2_pt3, cropped_tr2_mask, points2, rect2 = triangulation_face2(landmarks_points2, triangle_index, im_source)
#     (x, y, w, h) = rect2
#
#     # Warp triangles
#     points = np.float32(points)
#     points2 = np.float32(points2)
#     M = cv2.getAffineTransform(points, points2)
#     warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
#     warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)
#
#     # Reconstructing destination face
#     img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
#     img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
#     _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
#     warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)
#
#     img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
#     img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
#
# for filename in glob.glob('*.jpg'):
#     im = cv2.imread(filename, cv2.IMREAD_COLOR)
#     im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
#                          im.shape[0] * SCALE_FACTOR))
#     rects = detector(im, 1)
#     if len(rects) < 2:
#         print(filename + " is missing two faces. skipping.")
#         continue
#
#     if rects[0].left() < rects[1].left():
#         im1, landmarks1 = (im, numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]))
#         im2, landmarks2 = (im, numpy.matrix([[p.x, p.y] for p in predictor(im, rects[1]).parts()]))
#     else:
#         im1, landmarks1 = (im, numpy.matrix([[p.x, p.y] for p in predictor(im, rects[1]).parts()]))
#         im2, landmarks2 = (im, numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()]))
#
#     landmarks_points = landmarks1.tolist()
#     points = np.array(landmarks_points, np.int32)
#     convexhull = cv2.convexHull(points)
#     triangles = delaunay_triangulation(convexhull, landmarks_points)
#     indexes_triangles = []
#     get_indexes_triangles(indexes_triangles, triangles, points)
#
#     img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     height, width, channels = im.shape
#
#     img2_new_face = np.zeros((height, width, channels), np.uint8)
#
#     landmarks_points2 = landmarks2.tolist()
#     points2 = np.array(landmarks_points2, np.int32)
#     convexhull2 = cv2.convexHull(points2)
#
#     lines_space_mask = np.zeros_like(img_gray)
#     lines_space_new_face = np.zeros_like(im)
#
#     for triangle_index in indexes_triangles:
#         recon_face(landmarks_points, landmarks_points2, triangle_index, img2_new_face, im, lines_space_mask)
#
#     img2_face_mask = np.zeros_like(img_gray)
#     img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
#     img2_face_mask = cv2.bitwise_not(img2_head_mask)
#
#     img2_head_noface = cv2.bitwise_and(im, im, mask=img2_face_mask)
#     result = cv2.add(img2_head_noface, img2_new_face)
#
#     (x, y, w, h) = cv2.boundingRect(convexhull2)
#     center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
#
#     seamlessclone = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
#
#     ##################
#
#     img1_new_face = np.zeros((height, width, channels), np.uint8)
#     for triangle_index in indexes_triangles:
#         recon_face(landmarks_points2, landmarks_points, triangle_index, img1_new_face, im, lines_space_mask)
#
#     img1_face_mask = np.zeros_like(img_gray)
#     img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
#     img1_face_mask = cv2.bitwise_not(img1_head_mask)
#     img2_head_noface = cv2.bitwise_and(im, im, mask=img1_face_mask)
#
#     result1 = cv2.add(img2_head_noface, img1_new_face)
#
#     (x, y, w, h) = cv2.boundingRect(convexhull)
#     center_face1 = (int((x + x + w) / 2), int((y + y + h) / 2))
#
#     seamlessclone1 = cv2.seamlessClone(result1, im, img1_head_mask, center_face1, cv2.NORMAL_CLONE)
#
#     img1_head_mask = cv2.cvtColor(img1_head_mask, cv2.COLOR_GRAY2BGR)
#     img2_head_mask = cv2.cvtColor(img2_head_mask, cv2.COLOR_GRAY2BGR)
#
#     right_head  = seamlessclone & img2_head_mask
#     left_head_no_right_head = cv2.subtract(seamlessclone1, img2_head_mask)
#     output_im = cv2.add(left_head_no_right_head, right_head)
#     # output_im = im1 * (1.0 - img1_head_mask) + seamlessclone1 * img1_head_mask
#     # output_im = output_im * (1.0 - img2_head_mask) + seamlessclone * img2_head_mask
#
#     cv2.imwrite('SingleVideoOutput/' + filename, output_im)
#     print(filename + " finished, adding.")

# video to image  ->>>>>>>>  ffmpeg -i testvideo1.mov output%05d.jpg
# image to video ->>>>>>>> ffmpeg -i output%05d.jpg out.mp4


import cv2
import dlib
import numpy
import ffmpeg

import sys
import glob
import shutil
import image_processing
import numpy as np


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
    ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                    RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)
    # ALIGN_POINTS = list(range(68))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    # video reader and writer
    cap = cv2.VideoCapture(rawVideo)

    #check if need rotate
    # rotateCode = check_rotation(rawVideo)

    # Initialize video writer for tracking video
    trackVideo = 'single_video_results/Output_' + rawVideo
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(3)), int(cap.get(4)) )
    # size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(trackVideo, fourcc, fps, size)
    frame_cnt = 0

    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        if np.shape(frame) == ():
            continue
        # if rotateCode is not None:
        #     frame = correct_rotation(frame, rotateCode)

    # rotate if iphone videos, comment if download videos
        frame = correct_rotation(frame, cv2.ROTATE_180)
    # ------------------------------------------------

        frame_cnt += 1


        im = frame.copy()
        im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,im.shape[0] * SCALE_FACTOR))
        rects = detector(im, 1)
        if len(rects) != 2:
            print("missing two faces. skipping.")
            continue

        if rects[0].left() < rects[1].left():
            im1, landmarks1 = (im, predictor(im, rects[0]))
            im2, landmarks2 = (im, predictor(im, rects[1]))
        else:
            im1, landmarks1 = (im, predictor(im, rects[1]))
            im2, landmarks2 = (im, predictor(im, rects[0]))

        landmarks_points = []
        for n in ALIGN_POINTS:
            x = landmarks1.part(n).x
            y = landmarks1.part(n).y
            landmarks_points.append((x, y))

        landmarks_points2 = []
        for n in ALIGN_POINTS:
            x = landmarks2.part(n).x
            y = landmarks2.part(n).y
            landmarks_points2.append((x, y))

        # landmarks_points = landmarks1.tolist()
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        triangles = delaunay_triangulation(convexhull, landmarks_points)
        indexes_triangles = []
        get_indexes_triangles(indexes_triangles, triangles, points)

        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        height, width, channels = im.shape

        img2_new_face = np.zeros((height, width, channels), np.uint8)

        # landmarks_points2 = landmarks2.tolist()
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

        lines_space_mask = np.zeros_like(img_gray)

        for triangle_index in indexes_triangles:
            recon_face(landmarks_points, landmarks_points2, triangle_index, img2_new_face, im, lines_space_mask)

        img2_face_mask = np.zeros_like(img_gray)
        img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
        img2_face_mask = cv2.bitwise_not(img2_head_mask)

        img2_head_noface = cv2.bitwise_and(im, im, mask=img2_face_mask)
        result = cv2.add(img2_head_noface, img2_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull2)
        center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone = cv2.seamlessClone(result, im, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

        ##################

        img1_new_face = np.zeros((height, width, channels), np.uint8)
        for triangle_index in indexes_triangles:
            recon_face(landmarks_points2, landmarks_points, triangle_index, img1_new_face, im, lines_space_mask)

        img1_face_mask = np.zeros_like(img_gray)
        img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
        img1_face_mask = cv2.bitwise_not(img1_head_mask)
        img2_head_noface = cv2.bitwise_and(im, im, mask=img1_face_mask)

        result1 = cv2.add(img2_head_noface, img1_new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull)
        center_face1 = (int((x + x + w) / 2), int((y + y + h) / 2))

        seamlessclone1 = cv2.seamlessClone(result1, im, img1_head_mask, center_face1, cv2.NORMAL_CLONE)

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
    target_video_path = 'single_video_test.mov'
    face_swap_in_a_single_video(target_video_path)
    print("face swap finished")