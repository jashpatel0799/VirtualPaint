import cv2
from cvzone.HandTrackingModule import HandDetector


# capture the image from live video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = HandDetector(detectionCon=0.8, maxHands=3)

while True:
    # read image from cap
    success, img = cap.read()
    
    hands, img = detector.findHands(img)
    # print(hands)
    # print(img)
    
    if hands:
        hand_1 = hands[0]
        lm_list_1 = hand_1["lmList"]    # list of hand landmarks
        bbox_1 = hand_1["bbox"]     # bounding box in x,y,w,h 
        centerPoint_1 = hand_1["center"]    # center of c_x, c_y
        hand_type_1 = hand_1["type"]   # Hand type left or right
        
        finger_1 = detector.fingersUp(hand_1)
        # length, info, img = detector.findDistance(lm_list_1[8], lm_list_1[12], img)
        # test = detector.findDistance(lm_list_1[8][:2], lm_list_1[12][:2], img)
        # print(test)
        # length, info, img = detector.findDistance(lm_list_1[8][:2], lm_list_1[12][:2], img)
        
        # print(bbox_1)
        
        
    if len(hands) == 2:
        hand_2 = hands[1]
        lm_list_2 = hand_2["lmList"] # list of hand landmarks
        bbox_2 = hand_2["bbox"]  # bounding box in x,y,w,h 
        centerPoint_2 = hand_2["center"]   # center of c_x, c_y
        hand_type_2 = hand_2["type"]   # Hand type left or right
        
        finger_2 = detector.fingersUp(hand_2)
        
        
        # print(hand_type_1, hand_type_2)
        # print(finger_1, finger_2)
        # length, info = detector.findDistance(lm_list_1[8], lm_list_2[8])
        length, info, img = detector.findDistance(lm_list_1[8][:2], lm_list_2[8][:2], img)
    
    cv2.imshow("Image",img)
    # cv2.waitKey(1)
    
    # To close frame
    if cv2.waitKey(1) & 0xFF == ord('x'):
        # cap.release()
        # cv2.destroyAllWindows()
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


