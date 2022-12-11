import cv2
import numpy as np
import os

import HandTracker as hndt


#############################################

brush_thikness = 7
easer_thikness = 50

#############################################

f_path = "Header"
p_list = os.listdir(f_path)
p_list.sort()
# print(p_list)

overlap_image = []

for c_path in p_list:
    op_image = cv2.imread(f'{f_path}/{c_path}')
    overlap_image.append(op_image)

# print(len(overlap_image))

img_header = overlap_image[0]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

draw_color = (0, 255, 255)

detector = hndt.handsDetector(detect_conf=0.85)

x_p, y_p = 0,0
img_canvas = np.zeros((720, 1280, 3),  np.uint8)
# img_canvas = np.random.randint(low=255, high=256, size=(720, 1280, 3), dtype=np.uint8)
# print(img_canvas.shape)

while True:
    # 1) import image
    success, img = cap.read()
    # print(img.shape)
    img = cv2.flip(img, 1)

    # 2) Find Hands Landmarks
    img = detector.find_hands(img, draw=True)
    lm_list = detector.find_hands_position(img, draw=True)

    if len(lm_list)!=0:
        # print(lm_list)
        # tip of index and middle finger
        x_1, y_1 = lm_list[8][1:] 
        x_2, y_2 = lm_list[12][1:]

        
        # 3) Check which fingerup
        fingers = detector.fingers_up()
        # print(fingers)

        # 4) if selection mode - two finger (index and middle one)
        if fingers[1] and fingers[2]:
            x_p, y_p = 0,0

            cv2.rectangle(img, (x_1, y_1-30), (x_2, y_2+30), draw_color, cv2.FILLED)
            # print("selection mode")
            # checking for click
            if y_1 < 109:
                if 195<x_1<235:
                    img_header = overlap_image[0]
                    draw_color = (0, 255, 255)

                elif 340<x_1<380:
                    img_header = overlap_image[1]
                    draw_color = (0, 0, 255)

                elif 480<x_1<510:
                    img_header = overlap_image[2]
                    draw_color = (0, 255, 0)

                elif 620<x_1<650:
                    img_header = overlap_image[3]
                    draw_color = (255, 0, 0)

                elif 760<x_1<790:
                    img_header = overlap_image[4]
                    draw_color = (255, 255, 255)

                elif 820<x_1<890:
                    img_header = overlap_image[5]
                    draw_color = (0, 0, 0)


        # 5) if drawing mode - one finger (index one)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x_1, y_1), 15, draw_color, cv2.FILLED)
            # print("drawing mode")
            if x_p == 0 and y_p == 0:
                x_p, y_p = x_1, y_1


            if draw_color == (0, 0, 0):
                cv2.line(img, (x_p, y_p), (x_1, y_1), draw_color, easer_thikness)
                cv2.line(img_canvas, (x_p, y_p), (x_1, y_1), draw_color, easer_thikness)

            else:
                cv2.line(img, (x_p, y_p), (x_1, y_1), draw_color, brush_thikness)
                cv2.line(img_canvas, (x_p, y_p), (x_1, y_1), draw_color, brush_thikness)

            x_p, y_p = x_1, y_1

    img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 25, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)


    # img = cv2.bitwise_and(img, img)
    img = cv2.bitwise_and(img, img_inv)
    # img = cv2.bitwise_or(img, img)
    img = cv2.bitwise_or(img, img_canvas)


    # setting the header
    # if img_header != overlap_image[5]:
    #     img[0:109, 0:1000] = img_header
    # else:
    img[0:109, 0:1000] = img_header


    # img = cv2.addWeighted(img, 0.5, img_canvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", img_canvas)
    cv2.imshow("inv", img_inv)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()