import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class KeyMatcher:
    key_template = cv2.imread('src/image_template/keys2.jpg')

    def __init__(self):
        self.min_keys = 16
        self.max_keys = 8*7 # 8 octaves, 7 white keys per octave
        self.possible_zoom = 1.3
        self.img_scaled_width = 360
        self.min_match = 0.8
        self.n_template_scale_factors = 20
        self.best_scale_factor = None

    def ContainsKeyboard(self, img) -> bool:
        scaled_img, scale_factor = self.ScaleImage(img)
        matches, (top, bottom) = self.GetBestTemplateMatch(scaled_img)
        contains = matches.max() >= self.min_match
        return contains

    def ScaleImage(self, img) -> tuple[np.ndarray, float]:
        scale_factor = self.img_scaled_width / img.shape[1]
        scaled_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
        return scaled_img, scale_factor
    
    def GetBestScaleFactor(self, img) -> float:
        if self.best_scale_factor is not None:
            return self.best_scale_factor

        img_min_key_width = img.shape[1] / (self.max_keys*self.possible_zoom)
        template_key_width = KeyMatcher.key_template.shape[1] / 3
        k_min = img_min_key_width / template_key_width

        img_max_key_width = img.shape[1] / self.min_keys
        max_factor = min(img.shape[0] / KeyMatcher.key_template.shape[0], img.shape[1] / KeyMatcher.key_template.shape[1] )
        k_max = min(img_max_key_width / template_key_width, max_factor)

        best = None
        factors = np.linspace(k_min, k_max, self.n_template_scale_factors)
        for factor in factors:
            ckeys = cv2.resize(KeyMatcher.key_template, (0, 0), fx=factor, fy=factor)

            res = cv2.matchTemplate(img, ckeys, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if best is None or max_val > best[0]:
                best = (max_val, max_loc, factor)

        if best[0] > self.min_match:
            self.best_scale_factor = best[2]
        
        return best[2]
    
    def GetBestTemplateMatch(self, img) -> np.ndarray:
        template_scale = self.GetBestScaleFactor(img)

        ckeys = cv2.resize(KeyMatcher.key_template, (0, 0), fx=template_scale, fy=template_scale)
        res = cv2.matchTemplate(img, ckeys, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        top = max(0, int(max_loc[1] - ckeys.shape[0]*0.85)) # keyboard top and bottom
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