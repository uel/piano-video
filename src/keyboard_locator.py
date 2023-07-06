import cv2 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

class KeyboardLocator:
    def __init__(self):
        self.key_template = cv2.imread('template/keys2.jpg')
        self.min_keys = 16
        self.max_keys = 8*7 # 8 octaves, 7 white keys per octave
        self.img_scaled_width = 360
        self.min_match = 0.85
        self.n_template_scale_factors = 50

    def LocateKeyboard(self, path) -> np.ndarray:
        #path = 'data/separated_frames/with_keyboard/lwZg2ve_mz4_65.jpg'
        img = cv2.imread(path)
        scaled_img, scale_factor = self.ScaleImage(img)
        matches, (top, bottom) = self.GetBestTemplateMatch(scaled_img)
        # clusters = self.GetClusters(matches)


        
        # img_with_clusters = scaled_img.copy()
        # for cluster in clusters:
        #     cv2.circle(img_with_clusters, (cluster[1], cluster[0]), 2, (0, 0, 255), -1)

        # cv2.line(img_with_clusters, (0, top), (img_with_clusters.shape[1], top), (0, 255, 0), 1)
        # cv2.line(img_with_clusters, (0, bottom), (img_with_clusters.shape[1], bottom), (0, 255, 0), 1)
        
        # plt.ioff()
        # plt.figure(figsize=(20, 10))
        # plt.subplot(211)
        # plt.imshow(img_with_clusters[:, :, ::-1])
        # plt.subplot(212)
        # plt.imshow(matches)
        # plt.savefig('data/keyboard_locator/matches/'+os.path.basename(path))
        # plt.close()

        # top, right, bottom, left = self.RemoveAnimated(img)
        # scaled_image = img[top:bottom, left:right]
        # try:
        #     cv2.imwrite('data/keyboard_locator/remove_animated/'+os.path.basename(path), scaled_image)
        # except Exception as e:
        #     print(e)

        # plt.figure(figsize=(20, 10))
        # plt.subplot(211)
        # plt.imshow(scaled_img[:, :, ::-1])
        # plt.subplot(212)
        # plt.imshow(matches)
        # plt.show()



        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()
        # bounding_box = self.GetBoundingBox(clusters, scaled_img) # Find likely line around keyboard if it exists
        # bounding_box *= scale_factor
        # bounding_box += np.array([left, bottom])
        # return bounding_box
        return (0, int(top/scale_factor)), (img.shape[1], int(top/scale_factor)), (img.shape[1], int(bottom/scale_factor)), (0, int(bottom/scale_factor))
    

    def RemoveAnimated(self, original_img) -> tuple[int, int, int, int]:
        # Get border values where count of colors is larger that x% of width/height
        def GetBorder(img, dim, reverse, threshold):
            r =  reversed(range(0, img.shape[dim])) if reverse else range(0, img.shape[dim])
            for i in r:
                unique_colors = np.unique(img[i] if not dim else img[:,i], axis=0, return_counts=True)
                if unique_colors[1].shape[0] > threshold * img.shape[1-dim]:
                    return i
            return -1

        img = original_img.copy()
        top = GetBorder(img, 0, False, 0.25)
        bottom = GetBorder(img, 0, True, 0.25)
        img = img[top:bottom]
        
        #left = GetBorder(img, 1, False, 0.2)
        #right = GetBorder(img, 1, True, 0.2)

        left = 0
        right = img.shape[1]

        # print(top, right, bottom, left)

        # plt.figure(figsize=(12, 8))
        # plt.subplot(121)
        # plt.imshow(original_img)
        # plt.subplot(122)
        # plt.imshow(original_img[top:bottom, left:right])
        # plt.show()

        return top, right, bottom, left

    def ScaleImage(self, img) -> tuple[np.ndarray, float]:
        scale_factor = self.img_scaled_width / img.shape[1]
        scaled_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        return scaled_img, scale_factor

    def GetBestTemplateMatch(self, img) -> np.ndarray:
        img_min_key_width = img.shape[1] / self.max_keys
        template_key_width = self.key_template.shape[1] / 3
        k_min = img_min_key_width / template_key_width

        img_max_key_width = img.shape[1] / self.min_keys
        max_factor = min(img.shape[0] / self.key_template.shape[0], img.shape[1] / self.key_template.shape[1] )
        k_max = min(img_max_key_width / template_key_width, max_factor)

        best = None
        factors = np.linspace(k_min, k_max, self.n_template_scale_factors)
        for factor in factors:
            ckeys = cv2.resize(self.key_template, (0, 0), fx=factor, fy=factor)

            res = cv2.matchTemplate(img, ckeys, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if best is None or max_val > best[0]:
                best = (max_val, max_loc, factor)
        
        ckeys = cv2.resize(self.key_template, (0, 0), fx=best[2], fy=best[2])
        res = cv2.matchTemplate(img, ckeys, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top = max(0, int(max_loc[1] - ckeys.shape[0]*0.85))
        bottom = min(img.shape[0],  int(max_loc[1] + ckeys.shape[0]*1.3))
        
        return res, (top, bottom)

    def GetClusters(self, matches):  
        coords = np.array(np.where(matches >= self.min_match)).T

        best_cluster_count = (-1, 0)
        for i in range(2, min(8, coords.shape[0])):
            kmeans = KMeans(n_clusters=i, random_state=0, n_init=1).fit(coords)
            score = silhouette_score(coords, kmeans.labels_)
            if score > best_cluster_count[0]:
                best_cluster_count = (score, i)

        kmeans = KMeans(n_clusters=best_cluster_count[1], random_state=0, n_init=1).fit(coords)
        # Get max value and location  from each cluster
        max_points = []
        best_match = (-1, None)
        for i in range(best_cluster_count[1]):
            cluster = coords[kmeans.labels_ == i]
            max_idx = np.argmax(matches[cluster[:, 0], cluster[:, 1]])
            if best_match[0] < matches[cluster[max_idx][0], cluster[max_idx][1]]:
                best_match = (matches[cluster[max_idx][0], cluster[max_idx][1]], cluster[max_idx])
            max_points.append(cluster[max_idx])

        max_points = np.array(max_points)

        # check if matched at multiple y values
        for point in max_points[1:]:
            if abs(point[0]-max_points[0][0]) > 0.1 * matches.shape[0]:
                return np.array([best_match[1]])

        # remove points that are too far from the median y value
        #median_dev = np.median(np.abs(max_points[:, 0] - np.median(max_points[:, 0])))

        #median_y = np.median(max_points[:, 0])
        #max_points = max_points[np.abs(max_points[:, 0] - median_y) < 0.1 * matches.shape[0]]
        
        #display points 
        # for point in max_points:
        #     cv2.circle(img, (point[1], point[0]), 1, (0, 0, 255), -1)
        # plt.imshow(img[:, :, ::-1])
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()

        return max_points

class KeyboardLocatorNN:
    pass
   

# Detect keyboard using NN
# Find intervals in video
# Average/median of frames in intervals
# Find keyboard in averaged frames
# Segment keyboard into keys

# 1. Find any occurrence using ORB
# 2. Validate using template matching
# 3. Get bounding box TODO: get exact bounding box
# 5. Crop and de-skew, fish eye correction
# 6. Sparse linear + Binary search using ORB in cropped video to find all sub-segments times
# 7. Get vertical lines on average of cropped video
# 8. Use vertical lines to get contours of each key, label each key 