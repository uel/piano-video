import os
import cv2 
import numpy as np

orb = cv2.ORB_create()


def DetectKeyboardOrb(img):
    keys = cv2.imread('template/keys.png')

    # extract features using ORB
    kp, des = orb.detectAndCompute(keys, None)

    # extract features using ORB
    kp2, des2 = orb.detectAndCompute(img, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(des, des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance) 

    coords = np.array([kp2[m.trainIdx].pt for m in matches]).astype(int)

    avg_dist = sum([m.distance for m in matches]) / len(matches)

    return coords, avg_dist

def DetectKeyboardTemplateMatching(img):
    keys = cv2.imread('template/keys2.jpg')

    # template is 3 white keys wide
   
    min_keys = 16
    max_keys = 8*8

    img_min_key_width = img.shape[1] / max_keys
    template_key_width = keys.shape[1] / 3
    k = img_min_key_width / template_key_width
    
    keys = cv2.resize(keys, (0, 0), fx=k, fy=k)

    factors = np.linspace(1, (img.shape[1]/min_keys)/img_min_key_width, 50)

    best = None

    for factor in factors:
        ckeys = cv2.resize(keys, (0, 0), fx=factor, fy=factor)
        res = cv2.matchTemplate(img, ckeys, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if best is None or max_val > best[0]:
            best = (max_val, max_loc, factor)
    
    max_val, max_loc, factor = best

    top_left = max_loc
    bottom_right = (top_left[0] + int(keys.shape[1]*factor), top_left[1] + int(keys.shape[0]*factor))

    return top_left, bottom_right, max_val


def GetPianoBoundingBox(img):
    # location of keys
    top_left, bottom_right, val = DetectKeyboardTemplateMatching(img)
    keyboard_center_y = (top_left[1] + bottom_right[1]) // 2


    # get mask of white keys
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white_key_mask = cv2.inRange(img_hsv, (0, 0, 150), (255,  105, 255)) // 255
    # convert to black and white

    # sum rows of mask
    row_sums = np.sum(white_key_mask, axis=1)
    width = white_key_mask.shape[1]

    # find first row with less than 20% white pixels under keyboard center
    bottom_y = 0
    for i in range(keyboard_center_y, white_key_mask.shape[0]):
        if row_sums[i] / width < 0.2 and i > keyboard_center_y:
            bottom_y = i
            break
    
    # find first row with less than 10% white pixels above keyboard center
    top_y = 0
    for i in range(keyboard_center_y, 0, -1):
        if row_sums[i] / width < 0.1 and i < keyboard_center_y:
            top_y = i
            break

    white_key_mask = cv2.inRange(img_hsv, (0, 0, 0), (255,  155, 65)) # dont allow high saturation
    white_key_mask = cv2.blur(white_key_mask, (3, 3))
    white_key_mask = white_key_mask // 255
    mask_cropped = white_key_mask[top_y+int(0.8*(bottom_y-top_y)):bottom_y, :] # only white keys


    column_sums = np.sum(mask_cropped, axis=0)
    left_x = 0
    for i in range(mask_cropped.shape[1]//2, 0, -1):
        if column_sums[i] / (0.8*mask_cropped.shape[0]) > 0.4:
            left_x = i
            break
    
    right_x = mask_cropped.shape[1]
    for i in range(mask_cropped.shape[1]//2, mask_cropped.shape[1]):
        if column_sums[i] / (0.8*mask_cropped.shape[0]) > 0.4:
            right_x = i
            break
    
    return (left_x, top_y), (right_x, bottom_y)

def GetWhiteKeyLines(img, bounding_box):
    img_cropped = img[bounding_box[0][1]:bounding_box[1][1], bounding_box[0][0]:bounding_box[1][0]]
    # only white keys
    img_cropped = img_cropped[int(0.7*img_cropped.shape[0]):-10, :]

    lines = cv2.Canny(img_cropped, 50, 30)
    # lines = cv2.blur(lines, (3, 3))
    # lines[lines > 120] = 255
    # lines[lines <= 120] = 0

    cv2.imshow('lines', lines)
    cv2.waitKey(0)

    lines = cv2.HoughLinesP(lines, 1, np.pi/180, 1, minLineLength=img_cropped.shape[0]//2, maxLineGap=5)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_cropped, (x1, y1), (x2, y2), (0, 0, 255), 2)

    
    cv2.imshow('lines', img_cropped)
    cv2.waitKey(0)

    pass



# 1. Find any occurrence using ORB
# 2. Validate using template matching
# 3. Get bounding box TODO: get exact bounding box
# 5. Crop and de-skew, fish eye correction
# 6. Sparse linear + Binary search using ORB in cropped video to find all sub-segments times
# 7. Get vertical lines on average of cropped video
# 8. Use vertical lines to get contours of each key, label each key 

# folder = 'data/handless_keyboard_frames/'
# for filename in os.listdir(folder):
#     img = cv2.imread(folder + filename)

#     # resize so the width is 480 pixels
#     img = cv2.resize(img, (720, int(img.shape[0] * (720 / img.shape[1]))))


#     # matches, avg_dist = DetectKeyboardOrb(img)
#     # for match in matches:
#     #     cv2.circle(img, match, 3, (0,0,255), 2)

#     # print(avg_dist, end=' ')

#     # top_left, bottom_right, val = DetectKeyboardTemplateMatching(img)
#     # cv2.rectangle(img, top_left, bottom_right, (0,0,255), 2)
#     # print(val)


#     c1, c2 = GetPianoBoundingBox(img)
#     GetWhiteKeyLines(cv2.rectangle(img, c1, c2, (0,0,255), 2), (c1, c2))

#     cv2.imshow(filename, img)


#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
