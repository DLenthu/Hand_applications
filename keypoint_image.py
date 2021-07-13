import cv2
import mediapipe as mp

img = cv2.imread("Hand.jpg")
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
results = hands.process(imgRGB)
# print(results.multi_hand_landmarks)
if results.multi_hand_landmarks:
    for handlms in results.multi_hand_landmarks:
        for id ,lm in enumerate(handlms.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            # print(id,cx,cy)
            if id >= 0:
                cv2.circle(img,(cx,cy),10,(0,0,255),cv2.FILLED)

        mpDraw.draw_landmarks(img,handlms,mpHands.HAND_CONNECTIONS)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()