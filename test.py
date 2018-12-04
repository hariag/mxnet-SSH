import os
import cv2
import sys
import numpy as np
import datetime
#sys.path.append('.')
from ssh_detector import SSHDetector
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

scales = [1200, 1600]
#scales = [600, 1200]
t = 2
detector = SSHDetector('./model/e2ef', 0)


def visusalize_detections(im, bboxes, plt_name='output', ext='.png', visualization_folder=None, thresh=0.5):
    """
    A function to visualize the detections
    :param im: The image
    :param bboxes: The bounding box detections
    :param plt_name: The name of the plot
    :param ext: The save extension (if visualization_folder is not None)
    :param visualization_folder: The folder to save the results
    :param thresh: The detections with a score less than thresh are not visualized
    """
    inds = np.where(bboxes[:, -1] >= thresh)[0]
    bboxes = bboxes[inds]
    fig, ax = plt.subplots(figsize=(12, 12))


    if im.shape[0] == 3:
        im_cp = im.copy()
        im_cp = im_cp.transpose((1, 2, 0))
        if im.min() < 0:
            pixel_means = cfg.PIXEL_MEANS
            im_cp = im_cp + pixel_means

        im = im_cp.astype(dtype=np.uint8)

    im = im[:, :, (2, 1, 0)]

    ax.imshow(im, aspect='equal')
    if bboxes.shape[0] != 0:

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, :]
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=(0, bbox[4], 0), linewidth=3)
            )
	   

    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if visualization_folder is not None:
        if not os.path.exists(visualization_folder):
            os.makedirs(visualization_folder)
        plt_name += ext
        plt.savefig(os.path.join(visualization_folder,plt_name),bbox_inches='tight')
        print('Saved {}'.format(os.path.join(visualization_folder, plt_name)))
    else:
        print('Visualizing {}!'.format(plt_name))
        plt.show()
    plt.clf()
    plt.cla()


f = 'demo.jpg'
if len(sys.argv)>1:
  f = sys.argv[1]
img = cv2.imread(f)
im_shape = img.shape
print(im_shape)
target_size = scales[0]
max_size = scales[1]
im_size_min = np.min(im_shape[0:2])
im_size_max = np.max(im_shape[0:2])
if im_size_min>target_size or im_size_max>max_size:
  im_scale = float(target_size) / float(im_size_min)
  # prevent bigger axis from being more than max_size:
  if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)
  img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale)
  print('resize to', img.shape)
for i in xrange(t-1): #warmup
  faces = detector.detect(img)
timea = datetime.datetime.now()
faces = detector.detect(img, threshold=0.5)
timeb = datetime.datetime.now()
diff = timeb - timea
print('detection uses', diff.total_seconds(), 'seconds')
print('find', faces.shape[0], 'faces')
visusalize_detections(img, faces, visualization_folder=".")
