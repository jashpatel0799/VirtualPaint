import cv2
import mediapipe as mp
import time


# # capture the image from live video
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# # assign in built hand object to 
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands()
# mp_draw = mp.solutions.drawing_utils


# curr_time = 0
# prev_time = 0

# while True:
#     # read image from cap
#     success, img = cap.read()
    
#     # convert image BGR to RGB
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # store RGB image result
#     results = hands.process(imgRGB)
    
#     ## detect multiple hands landmark
#     #print(results.multi_hand_landmarks)
    
#     if results.multi_hand_landmarks:
#         for hands_lm in results.multi_hand_landmarks:
#             for id, lm in enumerate(hands_lm.landmark):
#                 h, w, c = img.shape
#                 c_x, c_y = int(lm.x * w), int(lm.y * h)
                
#                 if id == 8:
#                     cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), cv2.FILLED)
                
#             # draw land mark of plam
#             mp_draw.draw_landmarks(img, hands_lm, mp_hands.HAND_CONNECTIONS)
#             # mp_draw.draw_landmarks(img, hands_lm)
    
#     curr_time = time.time()
#     fps = 1/(curr_time - prev_time)
#     prev_time = curr_time
    
#     cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    
#     cv2.imshow("Image",img)
#     # cv2.waitKey(1)
    
#     # To close frame
#     if cv2.waitKey(1) & 0xFF == ord('x'):
#         # cap.release()
#         # cv2.destroyAllWindows()
#         break
        
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()



class handsDetector():
    def __init__(self, modes = False, max_hands = 2, model_complexity = 0, detect_conf = 0.5, track_conf = 0.5):
        self.modes = modes
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.tip_ids = [4, 8, 12, 16, 20]
        
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.modes, self.max_hands, self.model_complexity, self.detect_conf, self.track_conf)
        # self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw = True):
        # convert image BGR to RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # store RGB image result
        self.results = self.hands.process(imgRGB)

        ## detect multiple hands landmark
        #print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for hands_lm in self.results.multi_hand_landmarks:
                if draw:
                    for id, lm in enumerate(hands_lm.landmark):
                        h, w, c = img.shape
                        c_x, c_y = int(lm.x * w), int(lm.y * h)

                        if id == 8:
                            # pass
                            cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), cv2.FILLED)

                    # draw land mark of plam
                    self.mp_draw.draw_landmarks(img, hands_lm, self.mp_hands.HAND_CONNECTIONS)
                    # mp_draw.draw_landmarks(img, hands_lm)
        
        return img
    
    
    def find_hands_position(self, img, hand_no = 0, draw = True):
        self.lm_list = []
        
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[hand_no]
            
            for id, lm in enumerate(hands.landmark):
                h, w, c = img.shape
                c_x, c_y = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, c_x, c_y])
                
                if draw:
                    if id == 8:
                        # pass
                        cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), cv2.FILLED)
                    
        return self.lm_list


    def fingers_up(self):
        fingers = []

        # Thump
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)

        else:
            fingers.append(0)


        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)

            else:
                fingers.append(0)

        return fingers

def main():
    curr_time = 0
    prev_time = 0
    
    # capture the image from live video
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    detector = handsDetector()
    
    while True:
        # read image from cap
        success, img = cap.read()
        img = detector.find_hands(img)
        lm_list = detector.find_hands_position(img)
        
        if len(lm_list) != 0:
            print(lm_list[8])
        
        curr_time = time.time()
        fps = 1/(curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

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
        


if __name__ == "__main__":
    main()
