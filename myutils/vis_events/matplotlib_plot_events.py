from itertools import repeat
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('seaborn-whitegrid')
import os
import matplotlib.animation as animation
from tqdm import tqdm
# import open3d as o3d


class event_visualisation():
    def plot_data(self, data: np.ndarray, path, is_save, DPI=300, cmap=None):
        H, W = data.shape[0:2]
        fig = plt.figure(figsize=(W / float(DPI), H / float(DPI)))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off') # remove white border
        ax.set_xticks([]);ax.set_yticks([])
        if cmap is not None:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(data)
        if is_save:
            assert path is not None
            fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def plot_frame(self, frame, is_save, path=None):
        """
        frame: np.ndarray, HxW
        """
        if len(frame.shape) == 2:
            self.plot_data(frame, path, is_save, cmap='gray')
        elif len(frame.shape) == 3:
            self.plot_data(frame, path, is_save)

    def plot_event_cnt(self, event_cnt, is_save, path=None, color_scheme="green_red", use_opencv=False, is_black_background=True):
        """
        event_cnt: np.ndarray, HxWx2, 0 for positive, 1 for negative

        'gray': white for positive, black for negative
        'green_red': green for positive, red for negative
        'blue_red': blue for positive, red for negative
        """
        assert color_scheme in ['green_red', 'gray', 'blue_red'], f'Not support {color_scheme}'

        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            if is_black_background:
                event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        elif color_scheme == "blue_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            if is_black_background:
                event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 1][mask_pos] = 0
            event_image[:, :, 0][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 1][mask_neg] = 0
            event_image[:, :, 0][mask_neg * mask_not_pos] = 0

        event_image = (event_image * 255).astype(np.uint8)
        if not use_opencv:
            event_image = cv2.cvtColor(event_image, cv2.COLOR_BGR2RGB)

        # self.plot_data(event_image, path, is_save)

        return event_image

    def plot_event_img(self, event_list, resolution, is_save, path=None):
        """
        event_list: np.ndarray, Nx4, [x, y, t, p], p:[-1, 1]
        resolution: list, [H,W]

        blue for positive, red for negative
        """
        x, y, p = event_list[:, 0], event_list[:, 1], event_list[:, 3]
        H, W = resolution[0], resolution[1]

        assert x.size == y.size == p.size
        assert H > 0
        assert W > 0

        x = x.astype('int')
        y = y.astype('int')
        img = np.full((H, W, 3), fill_value=255, dtype='uint8')
        mask = np.zeros((H, W), dtype='int32')
        p = p.astype('int')
        p[p == 0] = -1
        mask1 = (x >= 0) & (y >= 0) & (W >= x) & (H > y)
        mask[y[mask1], x[mask1]] = p
        img[mask == 0] = [255, 255, 255]
        img[mask == -1] = [255, 0, 0]
        img[mask == 1] = [0, 0, 255]

        self.plot_data(img, path, is_save)

        return img
