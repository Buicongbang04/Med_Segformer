import cv2, glob, numpy as np

paths = glob.glob("/home/bangbc/Documents/CapstoneProject/ThesisProject26_v2/work_dirs/brats20_b1/preds/preds_mask/*.png")

count_tumor = 0
for p in paths:
    m = cv2.imread(p, 0)
    u = np.unique(m)
    if 1 in u:
        print("Tumor mask:", p, "unique:", u, "tumor pixels:", (m == 1).sum())
        count_tumor += 1
        # if count_tumor >= 10:
        #     break

print("Total checked:", len(paths))
print("Tumor masks found:", count_tumor)