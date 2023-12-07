from draw import draw_landmarks_on_image
import math
import numpy as np

# given 4 flaot points of a axis-aligned rectangle and a query point, return the distance of the closest point on the rectangle to the query point
# return 0 if query point is inside the rectangle
def closest_point_to_rect(rect, query):
    x1, y1, x2, y2 = rect
    x, y = query

    if x1 <= x <= x2 and y1 <= y <= y2: return 0

    x_closest = max(x1, min(x, x2))
    y_closest = max(y1, min(y, y2))
    return math.sqrt((x_closest - x)**2 + (y_closest - y)**2)

def closest_finger(key, hands):
    finger_tips = []
    for hand, fingers in zip(["L", "R"], hands):
        for f_id, finger in enumerate(np.array(fingers)[[4, 8, 12, 16, 20]], 1): # indicies of finger tips
            finger_tips.append((hand+str(f_id), (finger[0], finger[1])))

    name, id, rect = key
    closest_finger = None
    closest_dist = math.inf
    for finger in finger_tips:
        dist = closest_point_to_rect(rect, finger[1])
        if dist < closest_dist:
            closest_dist = dist
            closest_finger = finger
    
    return closest_finger

