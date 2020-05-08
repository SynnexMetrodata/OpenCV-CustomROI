import cv2
import numpy 

ROIEnabled = False
ROIShape = []

def createROI(event, x, y, flag, param):
    global ROIEnabled, ROIShape

    if ROIEnabled:
        if event == cv2.EVENT_LBUTTONUP:
            ROIShape.append((x,y))

def drawROI(frame, roi):
    if not ROIEnabled:
        return
        
    start_point = (0,0)
    if roi:
        start_point = next(iter(roi))
    else:
        return
    zero_point = start_point    
    for item in roi:
        end_point = item
        cv2.line(frame, start_point, end_point, (0,255,0), 2)
        start_point = end_point
    cv2.line(frame, start_point, zero_point, (0,255,0), 2)
    
    point = numpy.array(roi)
    argma = point.argmax(axis=0)
    argmi = point.argmin(axis=0)
    (x1, y1) = (int(point[argmi[0]][0]), int(point[argmi[1]][1]))
    (x2, y2) = (int(point[argma[0]][0]), int(point[argma[1]][1]))
    print(x1, y1, x2, y2)    
    croppep = frame[y1:y2,x1:x2].copy()

    pts = point - point.min(axis=0)
    mask = numpy.zeros(croppep.shape[:2], numpy.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croppep, croppep, mask=mask)

    if croppep.shape[0] > 0 and croppep.shape[1] > 0:
        cv2.imshow("ROI", dst)
   
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("CustomROI")
    cv2.setMouseCallback("CustomROI", createROI)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            drawROI(frame, ROIShape)
            cv2.imshow("CustomROI", frame)

            key = cv2.waitKey(1) & 0xFF 
            if key == ord('q'):
                break
            if key == ord('r'):                
                ROIEnabled = not ROIEnabled            

            if key == ord('n'):
                ROIShape = []
            
    cap.release()
    cv2.destroyAllWindows()
    

