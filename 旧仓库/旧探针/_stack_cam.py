# -*- coding: utf-8 -*-
"""把两个模型对同一个样本的注意力面板上下拼成一张对比图。"""
import os
import cv2
import numpy as np

HERE = r"D:\deeplearning\make_dataset\paoku"
out = os.path.join(HERE, "runs", "_camCmp")
os.makedirs(out, exist_ok=True)

ids = ["A_000000", "D_000000", "S_000000"]
for i in ids:
    head = i.split("_")[0]
    pa = os.path.join(HERE, "runs", "_camA", "attention_%s_%s.png" % (head, i))
    pb = os.path.join(HERE, "runs", "_camB", "attention_%s_%s.png" % (head, i))
    A = cv2.imread(pa); B = cv2.imread(pb)
    print(i, A.shape, B.shape)
    w = max(A.shape[1], B.shape[1])

    def pad(x):
        if x.shape[1] == w:
            return x
        return np.concatenate([x, np.full((x.shape[0], w - x.shape[1], 3), 22, np.uint8)], axis=1)

    A, B = pad(A), pad(B)

    def bar(txt, col):
        b = np.full((34, w, 3), 30, np.uint8)
        cv2.putText(b, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
        return b

    img = np.concatenate([bar("A INITIAL   chudai", (80, 220, 255)), A,
                          bar("B ITERATION diedai", (120, 255, 120)), B], axis=0)
    p = os.path.join(out, "cmp_%s.png" % i)
    cv2.imwrite(p, img)
    print(" ->", p, img.shape)
