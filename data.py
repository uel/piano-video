import numpy as np
import xml.etree.ElementTree as ET

def Sort4Points(points):
    # sort points clockwise
    # https://stackoverflow.com/a/6989383
    centroid = np.mean(points, axis=0)
    points = sorted(points, key=lambda p: np.arctan2(p[1] - centroid[1], p[0] - centroid[0]))
    return points

def GetPointsFromXML(file):
    root = ET.parse(file).getroot()

    res = []
    images = []
    for image in root.findall("image"):
        points = image[0].attrib["points"]
        points = points.split(';')
        points = [point.split(',') for point in points]
        points = [(float(point[0]), float(point[1])) for point in points]
        points = [(int(point[0]), int(point[1])) for point in points]
        points = Sort4Points(points)
        res.append(points)
        images.append(image.attrib["name"])
    
    res = np.array(res).astype(np.int32)
    return res, images