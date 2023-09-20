# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
import xml.etree.ElementTree as ET

import SimpleITK as sitk
import cv2
import numpy as np

"""
date:2020/5/6
"""


def read_file(filepath):
    """
    read the information in the .annot (itk-snap annotation file)
    :param filepath: the directionary of your targeted file
    :return: RECIST: x11,y11,x12,y12,x21,y21,x22,y22
    slice_th: the slice of RECIST
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    recist = [[]]
    slice_th = 0
    slice_recist = []
    z_point = []
    count = 0
    for child in root:
        if child.tag == 'folder':
            for grandkid in child:
                if grandkid.tag == 'folder':
                    color = list(map(float, grandkid[0].attrib['value'].split(' ')))
                    if count == 0:
                        count = count + 1
                        coloruse = color
                    if color[0] == coloruse[0] and color[1] == coloruse[1] and color[2] == coloruse[2]:
                        try:
                            point1 = list(map(float, grandkid[2].attrib['value'].split(' ')))
                            point2 = list(map(float, grandkid[3].attrib['value'].split(' ')))
                            if (point1[0] != point2[0] or point1[1] != point2[1] or point1[-1] != point2[-1]):
                                if slice_th != 0 and point1[-1] == slice_th:
                                    recist[-1] += point1[0:2] + point2[0:2]
                                elif slice_th != 0:
                                    recist.append(point1[0:2] + point2[0:2])
                                    slice_th = point1[-1]
                                    slice_recist.append(int(slice_th))
                                else:
                                    slice_th = point1[-1]
                                    recist[0] = point1[0:2] + point2[0:2]
                                    slice_recist.append(int(slice_th))
                            else:
                                z_point.append(point1[-1])
                        except:
                            print('There is no point')
    # return recist,slice_recist
    return recist, min(z_point), max(z_point), slice_recist


def read_file1(filepath):
    """
    read the information in the .annot (itk-snap annotation file)
    :param filepath: the directionary of your targeted file
    :return: RECIST: x11,y11,x12,y12,x21,y21,x22,y22
    slice_th: the slice of RECIST
    """
    tree = ET.parse(filepath)
    root = tree.getroot()
    recist = [[]]
    slice_th = []
    slice_recist = []
    z_point = []
    count = 0
    for child in root:
        if child.tag == 'folder':
            for grandkid in child:
                if grandkid.tag == 'folder':
                    color = list(map(float, grandkid[0].attrib['value'].split(' ')))
                    if count == 0:
                        count = count + 1
                        coloruse = color
                    if color[0] == coloruse[0] and color[1] == coloruse[1] and color[2] == coloruse[2]:
                        try:
                            point1 = list(map(float, grandkid[2].attrib['value'].split(' ')))
                            point2 = list(map(float, grandkid[3].attrib['value'].split(' ')))
                            if (point1[0] != point2[0] or point1[1] != point2[1] or point1[-1] != point2[-1]):
                                slice_th_set = set(slice_th)
                                if len(slice_th) != 0 and point1[-1] in slice_th_set:
                                    p = slice_th.index(point1[-1])
                                    recist[p] += point1[0:2] + point2[0:2]
                                elif len(slice_th) != 0:
                                    recist.append(point1[0:2] + point2[0:2])
                                    slice_th.append(point1[-1])
                                    slice_recist.append(int(point1[-1]))
                                else:
                                    slice_th.append(point1[-1])
                                    recist[0] = point1[0:2] + point2[0:2]
                                    slice_recist.append(int(point1[-1]))
                            else:
                                z_point.append(point1[-1])
                        except:
                            print('There is no point')
    return recist, slice_recist
    # return recist,min(z_point),max(z_point),slice_recist


def gen_mask_polygon_from_recist(recist):
    """
    Generate ellipse from RECIST
    :param recist:
    :return:
    """
    assert len(recist) == 8
    x11, y11, x12, y12, x21, y21, x22, y22 = recist
    axis1 = np.linalg.solve(np.array([[x11, y11], [x12, y12]]), np.array([1, 1]))
    axis2 = np.linalg.solve(np.array([[x21, y21], [x22, y22]]), np.array([1, 1]))
    center = np.linalg.solve(np.array([[axis1[0], axis1[1]], [axis2[0], axis2[1]]]), np.array([1, 1]))
    center_recist = recist - np.tile(center, (4,))
    center_recist = np.reshape(center_recist, (4, 2))
    pt_angles = np.arctan2(center_recist[:, 1], center_recist[:, 0])
    pt_lens = np.sqrt(np.sum(center_recist ** 2, axis=1))
    ord = [0, 2, 1, 3, 0]
    grid = .1
    rotated_pts = []
    for p in range(4):
        if (pt_angles[ord[p]] < pt_angles[ord[p + 1]] and pt_angles[ord[p + 1]] - pt_angles[ord[p]] < np.pi) \
                or (pt_angles[ord[p]] - pt_angles[ord[p + 1]] > np.pi):
            angles = np.arange(0, np.pi / 2, grid)
        else:
            angles = np.arange(0, -np.pi / 2, -grid)
        xs = np.cos(angles) * pt_lens[ord[p]]
        ys = np.sin(angles) * pt_lens[ord[p + 1]]
        r = pt_angles[ord[p]]
        rotated_pts1 = np.matmul(np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]), np.vstack((xs, ys)))
        rotated_pts.append(rotated_pts1)
    rotated_pts = np.hstack(rotated_pts)
    decentered_pts = rotated_pts + center.reshape((2, 1))
    polygon = decentered_pts.transpose().ravel()
    return polygon.tolist()


def getcross(recist, z):
    """
    get the recist crossline as [(x1,y1,z1),...,(xn,yn,zn)]
    :param recist:
    :param z:
    :return:
    """
    x11, y11, x12, y12, x21, y21, x22, y22 = recist
    axis1 = np.linalg.solve(np.array([[x11, y11], [x12, y12]]), np.array([1, 1]))
    axis2 = np.linalg.solve(np.array([[x21, y21], [x22, y22]]), np.array([1, 1]))
    a1 = axis1[0]
    b1 = axis1[1]
    a2 = axis2[0]
    b2 = axis2[1]
    line = []

    z = int(z)
    if abs(a1 / b1) < 1:
        for x in range(int(min(x11, x12) + 1), int(max(x11, x12) - 1)):
            line.append((int(x), int(-a1 / b1 * x + 1 / b1), z))
            # for numpy array use '[]'
            # for simpleitk image change '[]' to '()'
    else:
        for y in range(int(min(y11, y12) + 1), int(max(y11, y12) - 1)):
            line.append((int(-b1 / a1 * y + 1 / a1), int(y), z))

    if abs(a2 / b2) < 1:
        for x in range(int(min(x21, x22) + 1), int(max(x21, x22) - 1)):
            line.append((int(x), int(-a2 / b2 * x + 1 / b2), z))
    else:
        for y in range(int(min(y21, y22) + 1), int(max(y21, y22) - 1)):
            line.append((int(-b2 / a2 * y + 1 / a2), int(y), z))
    return line


def get_polygon_and_line(recist, recist_slice):
    polygon_all = []
    line_all = []
    for i in range(len(recist)):
        polygon_tmp = gen_mask_polygon_from_recist(recist[i])
        line_tmp = getcross(recist[i], recist_slice[i])
        polygon_all.append(polygon_tmp)
        line_all.append(line_tmp)
    return polygon_all, line_all


def get_cross_from_annot(recist):
    """
    support multi-slices annotation and on each slice support multi-crosses
    :param recist: get from read_file1
    :return: cross_list [i][j] the jth cross on ith slice
    """
    cross_list = []
    for i in range(recist.__len__()):
        recist_sovle = recist[i]
        recist_sovle = np.asarray(recist_sovle)
        line_num = int(recist_sovle.__len__() / 4)
        recist_sovle = np.reshape(recist_sovle, (line_num, -1))

        total = 0
        for j in range(line_num):
            line1 = recist_sovle[j]
            for k in range(j + 1, line_num):
                line2 = recist_sovle[k]

                x1_min = min(line1[0], line1[2])
                x1_max = max(line1[0], line1[2])
                y1_min = min(line1[1], line1[3])
                y1_max = max(line1[1], line1[3])

                x2_min = min(line2[0], line2[2])
                x2_max = max(line2[0], line2[2])
                y2_min = min(line2[1], line2[3])
                y2_max = max(line2[1], line2[3])

                x_min = max(x1_min, x2_min)
                x_max = min(x1_max, x2_max)
                y_min = max(y1_min, y2_min)
                y_max = min(y1_max, y2_max)

                axis1 = np.linalg.solve(np.array([[line1[0], line1[1]], [line1[2], line1[3]]]), np.array([1, 1]))
                axis2 = np.linalg.solve(np.array([[line2[0], line2[1]], [line2[2], line2[3]]]), np.array([1, 1]))
                if max(abs(axis1[0] - axis2[0]), abs(axis1[1] - axis2[1])) < 0.001:  # singular value#
                    continue
                center = np.linalg.solve(np.array([[axis1[0], axis1[1]], [axis2[0], axis2[1]]]), np.array([1, 1]))

                if center[0] <= x_max + 0.01 and center[
                    0] >= x_min - 0.01:  # relaxing by 0.01 for vertical and horizontal lines
                    if center[1] <= y_max + 0.01 and center[1] >= y_min - 0.01:
                        # cross_list.append([line1[0],line1[1],line1[2],line1[3],line2[0],line2[1],line2[2],line2[3]])
                        if total == 0:
                            total = 1
                            cross_list.append(
                                [[line1[0], line1[1], line1[2], line1[3], line2[0], line2[1], line2[2], line2[3]]])
                        else:
                            cross_list[-1] += [
                                [line1[0], line1[1], line1[2], line1[3], line2[0], line2[1], line2[2], line2[3]]]
    return cross_list


def ImageResample(sitk_image, new_spacing=[1.0, 1.0, 3.0], is_label=False):
    '''
        sitk_image:
        new_spacing: x,y,z
        is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
        '''
    size = np.array(sitk_image.GetSize())
    spacing = np.array(sitk_image.GetSpacing())
    new_spacing = np.array(new_spacing)
    new_size = size * spacing / new_spacing
    new_spacing_refine = size * spacing / new_size
    new_spacing_refine = [float(s) for s in new_spacing_refine]
    new_size = [int(s) for s in new_size]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetSize(new_size)
    resample.SetOutputSpacing(new_spacing_refine)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        # resample.SetInterpolator(sitk.sitkNearestNeighbor)
        # resample.SetInterpolator(sitk.sitkBSpline)
        resample.SetInterpolator(sitk.sitkLinear)

    newimage = resample.Execute(sitk_image)
    return newimage


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
    yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        iou1 = 0
        iou2 = 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    # boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    # boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    boxAArea = abs(boxA[2] * boxA[3])
    boxBArea = abs(boxB[2] * boxB[3])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    # iou = interArea / float(boxAArea + boxBArea - interArea)
    iou1 = float(interArea) / float(boxAArea)
    iou2 = float(interArea) / float(boxBArea)
    # return the intersection over union value
    return iou1, iou2


def draw(rects, color, ocv):
    for r in rects:
        p1 = (r[0], r[1])
        p2 = (r[0] + r[2], r[1] + r[3])
        cv2.rectangle(ocv, p1, p2, color, 2)


if __name__ == '__main__':
    anno_path = '/data/sdd/user/processed_data/anno/000133_02_01_038-050.annot'
    a = read_file(anno_path)
    print('done')
