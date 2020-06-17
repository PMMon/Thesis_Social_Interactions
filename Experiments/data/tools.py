# -*-coding:utf-8-*-
# Author: Shen Shen
# Email: dslwz2002@163.com

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
import json
import random
import os


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def g(x):
    return np.max(x, 0)   # Keep compatiable with numpy in 1.14.0 version


# 计算点到线段的距离，并计算由点到与线段交点的单位向量
def distanceP2W(point, wall):
    p0 = np.array([wall[0],wall[1]])
    p1 = np.array([wall[2],wall[3]])


    d = p1-p0
    ymp0 = point-p0
    t = np.dot(d,ymp0) /np.dot(d,d)
    if t > 0.0 and t < 1.:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point)

    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
        npw = normalize(cross - point) * 0

    """
        dist = np.sqrt(np.dot(ymp0,ymp0))
        cross = p0 + t*d

    elif t >= 1.0:
        ymp1 = point-p1
        dist = np.sqrt(np.dot(ymp1,ymp1))
        cross = p0 + t*d
    """

    return dist, npw

def image_json(scene, json_path ,  scaling = 1):


    json_path = os.path.join( json_path, "{}_seg.json".format(scene))

    wall_labels = ["lawn", "building", "car","roundabout"]

    walls = []
    wall_points = []

    start_end_points = {}
    decisionZone = {}
    directionZone = {}

    nr_start_end = 0

    with open(json_path) as json_file:

        data = json.load( json_file)
        for p in data["shapes"]:
            label = p["label"]
            if label in wall_labels:


                points = np.array(p["points"]).astype(int)

                points = order_clockwise(points)
                for i in np.arange(len(points)):


                    j = (i+1)%len(points)

                    p1 = points[i]
                    p2 = points[j]

                    concat = np.concatenate((p1, p2))
                    walls.append(scaling*concat)


                wall_points.append([p*scaling for p in  points])
            elif "StartEndZone" in label:
                id = int(label.split("_")[-1])
                start_end_points[nr_start_end] = {"point" :scaling*np.array(p["points"]),
                                                    "id" : id}
                nr_start_end+=1
            elif "decisionZone" in label:
                id = int(label.split("_")[-1])
                decisionZone[id]= scaling* np.array(p["points"])

            elif "directionZone" in label:
                id = int(label.split("_")[-1])
                directionZone[id] = scaling*np.array(p["points"])


    return walls, wall_points, start_end_points, decisionZone, directionZone

# order points clockwise

def order_clockwise(point_array, orientation = np.array([1, 0])):
    center = np.mean( point_array, axis = 0)
    directions = point_array - center

    angles = []
    for d in directions:
        #if d[0] > 0:

        #sign = np.sign(d[0])
        t = np.arctan2(d[1], d[0])
        #t = np.dot( d, orientation)/ np.sqrt(np.dot(d, d))
        angles.append(t)
    point_array = [x for _, x in sorted(zip(angles, point_array))]

    return point_array

def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds

    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if (random_point.within(poly)):
            break

    return random_point

if __name__ == '__main__':
    image_json()
    # v1 = np.array([3.33,3.33])
    # print(worldCoord2ScreenCoord(v1, [1000,800],30))
    # v2 = np.array([23.31,3.33])
    # print(worldCoord2ScreenCoord(v2,[1000,800] ,30))
    # v3 = np.array([29.97,23.31])
    # print(worldCoord2ScreenCoord(v3, [1000,800],30))
    wall = [3.33, 3.33, 29.97, 3.33]
    print(distanceP2W(np.array([10.0,10.0]),wall))
    # print distanceP2W(np.array([0.5,2.0]),wall)
    # print distanceP2W(np.array([2.0,2.0]),wall)
