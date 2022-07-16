import cv2
import numpy as np
import blend
from aligner import Aligner


source_path = "./images/source1.jpg"
target_path = "./images/target1.jpg"
aligned_source_path = "./images/aligned_source.jpg"
aligned_mask_path = "./images/aligned_mask.jpg"

target = cv2.imread(target_path)
aligner = Aligner(source_path, target_path, aligned_source_path, aligned_mask_path)
aligner.draw_aligned_mask_and_source()

source = cv2.imread(aligned_source_path)
mask = cv2.imread(aligned_mask_path, 0)
mask2 = np.atleast_3d(mask).astype('uint8') / 255
mask2[mask2 != 1] = 0
mask2 = mask2[:, :, 0]
# print(mask)

# if mask2.shape != source.shape:
#     print("shape unmatched")
#     exit(0)

channels = source.shape[-1]
result_stack = [blend.blend(source[:, :, i], target[:, :, i], mask2) for i in range(channels)]
# Merge
result = cv2.merge(result_stack)
filename = "result.jpg"
cv2.imwrite(filename, result)
