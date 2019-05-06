import cv2
import numpy as np 

drawing = False
pt1_x , pt1_y = None , None

# draw free line
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(image,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=30)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(image,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=30)    
	
image = np.zeros((512,512,1), np.uint8)
cv2.namedWindow('Press esc when done')
cv2.setMouseCallback('Press esc when done',line_drawing)

while(1):
    cv2.imshow('Press esc when done',image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()

image = cv2.resize(image, (28, 28))
input_vector = np.array(image).reshape(-1, 28 * 28) / 255.0

