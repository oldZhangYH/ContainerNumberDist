import os
import re
from tkinter import messagebox
from matplotlib import pyplot as plt
import cv2

# def creatLabel(path):
#     data = os.listdir(path)  # 目录里的所有文件
#
#     f = open(path+"label.txt", "w+")
#     for i in data:
#         f.write(i[0:i.rfind(".")])
#
# creatLabel("I:/ContainerNumber/")

path = "I:/ContainerNumber/img/"
#
# global count
# count = 0
#
#
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     global count
#     if event == cv2.EVENT_LBUTTONDOWN:
#         xy = "%d,%d" % (x, y)
#         cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
#         cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
#                     1.0, (0, 0, 0), thickness=1)
#         count += 1
#         f = open("I:/ContainerNumber/label/coordinate.txt", "a")
#         if count % 4 == 0:
#             f.write(xy + " " + "\n")
#         else:
#             f.write(xy + " ")
#         cv2.imshow("image", img)
#         f.close()
#
#
# files = os.listdir(path)
# # files = files[257:]
# for i in files:
#     img = cv2.imread(path + i)
#     f = open("I:/ContainerNumber/label/coordinate.txt", "a")
#     f.write(i + " ")
#     f.close()
#     cv2.namedWindow("image")
#     cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
#
#     while 1:
#         cv2.imshow("image", img)
#         if cv2.waitKey(0) & 0xFF == 27:
#             break
#     cv2.destroyAllWindows()
files = os.listdir(path)
coors = open("I:/ContainerNumber/label/coordinate.txt", "r")
# for coor in coors:
#     coor=list(coor)
#     coor.pop(11)
#     coor.pop(11)
#     coor.pop(11)
#     coor.pop(11)
#     coor = ''.join(coor)
#     f = open("I:/ContainerNumber/label/sdads.txt", "a")
#     f.write(coor)
#
label = []

for coor in coors:
    start = 16
    court = 0
    for i in coor[16:]:
        if i != " ":
            court += 1
        else:
            label.append(coor[start:start + court])
            start += court
            court = 1
res = []
for str in label:

    temp = 0
    for i in str:
        temp += 1
        if i == ",":
            res.append(int(str[:temp - 1]))
            res.append(int(str[temp:]))
            break
label = res
x = []
y = []
for i in range(0, len(label), 2):
    x.append(label[i])
for i in range(1, len(label), 2):
    y.append(label[i])

court = 0
for i in range(0, len(x), 4):
    img = cv2.imread(path + files[court])
    plt.imshow(img)
    plt.plot((x[i], x[i + 1]), (y[i], y[i + 1]))
    plt.plot((x[i], x[i + 2]), (y[i], y[i + 2]))
    plt.plot((x[i + 1], x[i + 3]), (y[i + 1], y[i + 3]))
    plt.plot((x[i + 2], x[i + 3]), (y[i + 2], y[i + 3]))
    plt.savefig("I:/ContainerNumber/test/"+files[court])
    plt.show()
    court += 1

print("debug")
