# OpenCV for tracking/display
import cv2
import time

# When program is started
if __name__ == '__main__':
    # Are we finding motion or tracking
    status = 'motion'
    # How long have we been tracking
    idle_time = 0

    # Background for motion detection
    back = None
    # An MIL tracker for when we find motion
    tracker = cv2.TrackerMIL_create()

    # Webcam footage (or video)
    video = cv2.VideoCapture(0)

    # LOOP
    while True:
        # Check first frame
        ok, frame = video.read()

        # Grayscale footage
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Blur footage to prevent artifacts
        gray = cv2.GaussianBlur(gray,(21,21),0)

        # Check for background
        if back is None:
            # Set background to current frame
            back = gray

        if status == 'motion':
            # Difference between current frame and background
            frame_delta = cv2.absdiff(back,gray)
            # Create a threshold to exclude minute movements
            thresh = cv2.threshold(frame_delta,25,255,cv2.THRESH_BINARY)[1]

            #Dialate threshold to further reduce error
            thresh = cv2.dilate(thresh,None,iterations=2)
            # Check for contours in our threshold
            _,cnts,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


            # Check each contour
            if len(cnts) != 0:
                # If the contour is big enough

                # Set largest contour to first contour
                largest = 0

                # For each contour
                for i in range(len(cnts)):
                    # If this contour is larger than the largest
                    if i != 0 & int(cv2.contourArea(cnts[i])) > int(cv2.contourArea(cnts[largest])):
                        # This contour is the largest
                        largest = i

                if cv2.contourArea(cnts[largest]) > 1000:
                    # Create a bounding box for our contour
                    (x,y,w,h) = cv2.boundingRect(cnts[0])
                    # Convert from float to int, and scale up our boudning box
                    (x,y,w,h) = (int(x),int(y),int(w),int(h))
                    # Initialize tracker
                    bbox = (x,y,w,h)
                    ok = tracker.init(frame, bbox)
                    # Switch from finding motion to tracking
                    status = 'tracking'


        # If we are tracking
        if status == 'tracking':
            # Update our tracker
            ok, bbox = tracker.update(frame)
            # Create a visible rectangle for our viewing pleasure
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame,p1,p2,(0,0,255),10)


        # Show our webcam
        cv2.imshow("Camera",frame)


        # If we have been tracking for more than a few seconds
        if idle_time >= 30:
            # Reset to motion
            status = 'motion'
            # Reset timer
            idle_time = 0

            # Reset background, frame, and tracker
            back = None
            tracker = None
            ok = None

            # Recreate tracker
            tracker = cv2.TrackerMIL_create()


        # Incriment timer
        idle_time += 1

        # Check if we've quit
        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('Camera',0) == -1:
            break
#QUIT
video.release()
cv2.destroyAllWindows()