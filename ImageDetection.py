import cv2
import numpy as np

class ImageDetection:
    def __init__(self):
        pass

    def findRed(self, img):
        K = np.array([[2564.3186869, 0, 0], 
                  [0, 2569.70273111, 0], 
                  [0, 0, 1]])

        # Focal lengths (from the intrinsic matrix)
        f_x = K[0, 0]
        f_y = K[1, 1]
        
        real_radius = 10

        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Setting bounds on what red is
        lower_red = np.array([0, 50, 100])
        upper_red = np.array([10, 255, 255])

        # Just compiling all of the red
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Setting more bounds on what red is
        lower_red2 = np.array([170, 50, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        lower_pink = np.array([140, 50, 50])
        upper_pink = np.array([170, 255, 255])
        mask3 = cv2.inRange(hsv, lower_pink, upper_pink)

        # Combine the masks to cover the full red hue range
        mask = mask1 + mask2 + mask3

        # Removing Noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours2, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Approximate the contour to a polygon
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            
            # Compute the area of the contour to filter out small noise
            area = cv2.contourArea(contour)
            if area > 1000:  # Threshold for minimum area
                # Draw the contour on the image
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)

                # Compute the centroid of the contour using moments
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if (radius > 0):
                        Z = (f_x * real_radius) / radius
                        X = (cx * Z) / f_x
                        Y = (cy * Z) / f_y

                        cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img
    
    # Overall not my smartest implementation, too many unnecessary if statements
    def findContoursSelf(self, img):

        K = np.array([[2564.3186869, 0, 0], 
                  [0, 2569.70273111, 0], 
                  [0, 0, 1]])

        # Focal lengths (from the intrinsic matrix)
        f_x = K[0, 0]
        f_y = K[1, 1]

        real_radius = 10
        
        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to the grayscale image
        blur = cv2.GaussianBlur(gray, (3,3), 0)

        # Apply binary thresholding
        _, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY)

        _, invthresh = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY_INV)

        # Find contours using the inverted thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # edges = cv2.Canny(image=blur, threshold1=100, threshold2=200), tried edge detection but it didn't work well
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                shape_name = "Triangle"
            elif len(approx) == 4:
                # Check if it's a rectangle or trapezoid
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)
                if aspectRatio >= 0.95 and aspectRatio <= 1.05 and cv2.contourArea(contour) > 1000:
                    shape_name = "Rectangle"
                    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                    M = cv2.moments(contour)

                    # Compute the centroid of the contour using moments
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if (radius > 0):
                            Z = (f_x * real_radius) / radius
                            X = (cx * Z) / f_x
                            Y = (cy * Z) / f_y

                            cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    if cv2.contourArea(contour) > 1000:
                        shape_name = "Trapezoid"
                        cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                        M = cv2.moments(contour)

                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                            (x, y), radius = cv2.minEnclosingCircle(contour)
                            if (radius > 0):
                                Z = (f_x * real_radius) / radius
                                X = (cx * Z) / f_x
                                Y = (cy * Z) / f_y

                                cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 5 and cv2.contourArea(contour) > 1000:
                shape_name = "Pentagon"
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if (radius > 0):
                        Z = (f_x * real_radius) / radius
                        X = (cx * Z) / f_x
                        Y = (cy * Z) / f_y

                        cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            elif len(approx) == 6 and cv2.contourArea(contour) > 1000:
                shape_name = "Hexagon"
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
            else:
                if cv2.contourArea(contour) > 1000:
                    shape_name = "Circle"
                    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                    M = cv2.moments(contour)

                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if (radius > 0):
                            Z = (f_x * real_radius) / radius
                            X = (cx * Z) / f_x
                            Y = (cy * Z) / f_y

                            cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        return img
    
    def findBlue(self, img):

        K = np.array([[2564.3186869, 0, 0], 
                  [0, 2569.70273111, 0], 
                  [0, 0, 1]])

        # Focal lengths (from the intrinsic matrix)
        f_x = K[0, 0]
        f_y = K[1, 1]

        real_radius = 10

        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the lower and upper range for blue color in HSV
        lower_blue = np.array([100, 150, 50])   # Lower bound of hue, saturation, and value
        upper_blue = np.array([140, 255, 255])  # Upper bound of hue, saturation, and value

        lower_green = np.array([39, 115, 126])
        upper_green = np.array([59, 195, 206])

        # Apply these bounds to create a mask
        mask1 = cv2.inRange(hsv, lower_green, upper_green)


        # Create a mask for the blue color
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Perform morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel)
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:

            area = cv2.contourArea(contour)
            if area > 1000:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
            M = cv2.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if (radius > 0):
                    Z = (f_x * real_radius) / radius
                    X = (cx * Z) / f_x
                    Y = (cy * Z) / f_y

                    cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return img
    
    def findImage(self, img):
        img = self.findRed(img)
        img = self.findBlue(img)
        img = self.findContoursSelf(img)

        return img
    
    def find2Frames(self, path):
        count = 0
        vid = cv2.VideoCapture(path)
        while True:
            ret, frame = vid.read()

            if count == 0:
                frame1 = frame
            if count == 200:
                frame2 = frame
                break
            count += 1
        return (frame1, frame2)
    
    def subtract_frames(self, frame1, frame2):
        # Convert frames to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Compute the absolute difference between the two frames
        diff = cv2.absdiff(gray1, gray2)

        # Apply a binary threshold to the difference image
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return cleaned
    
    # This function is not used in the final implementation, but I was toying with passing a binary image, then drawing contours on the original image
    def findBinary(self, img, color):
        _, thresh = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour,True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 3:
                shape_name = "Triangle"
            elif len(approx) == 4:

                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)
                if aspectRatio >= 0.95 and aspectRatio <= 1.05 and cv2.contourArea(contour) > 1000:
                    shape_name = "Rectangle"
                    cv2.drawContours(color, [approx], 0, (0, 0, 255), 2)
                    M = cv2.moments(contour)

                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    if cv2.contourArea(contour) > 1000:
                        shape_name = "Trapezoid"
                        cv2.drawContours(color, [approx], 0, (0, 0, 255), 2)
                        M = cv2.moments(contour)

                        if M['m00'] != 0:
                            cx = int(M['m10'] / M['m00'])
                            cy = int(M['m01'] / M['m00'])
                            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            elif len(approx) == 5 and cv2.contourArea(contour) > 1000:
                shape_name = "Pentagon"
                cv2.drawContours(color, [approx], 0, (0, 0, 255), 2)
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            elif len(approx) == 6 and cv2.contourArea(contour) > 1000:
                shape_name = "Hexagon"
                cv2.drawContours(color, [approx], 0, (0, 0, 255), 2)
            else:
                if cv2.contourArea(contour) > 1000:
                    shape_name = "Circle"
                    cv2.drawContours(color, [approx], 0, (0, 0, 255), 2)
                    M = cv2.moments(contour)

                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)
            
        return color
    



    def findGradients(self, img, blackwhite):

        blackwhite = cv2.cvtColor(blackwhite, cv2.COLOR_BGR2GRAY)

        _, threshold = cv2.threshold(blackwhite, 160, 255, cv2.THRESH_BINARY_INV)

        cv2.imshow('Threshold', threshold)
        cv2.waitKey(0)
        
        # Find contours from the edges
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if area > 1000:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
            M = cv2.moments(contour)

            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
        
        return img
    


    def findContoursSelf2(self, img, blackwhite):

        # Intrinsic matrix from the camera calibration
        K = np.array([[2564.3186869, 0, 0], 
                    [0, 2569.70273111, 0], 
                    [0, 0, 1]])

        # Focal lengths
        f_x = K[0, 0]
        f_y = K[1, 1]

        real_radius = 10
        
        blackwhite = cv2.cvtColor(blackwhite, cv2.COLOR_BGR2GRAY)

        # Apply binary thresholding
        _, thresh = cv2.threshold(blackwhite, 120, 255, cv2.THRESH_BINARY)
        
        # Apply inverse binary thresholding
        _, invthresh = cv2.threshold(blackwhite, 170, 255, cv2.THRESH_BINARY_INV)

        # Find contours using the inverted thresholded image
        contours, _ = cv2.findContours(invthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours to identify shapes and draw them
        for contour in contours:
            epsilon = 0.001 * cv2.arcLength(contour, True)  # Contour approximation
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)

            # Only consider contours with a significant area
            if area > 20000:
                # Draw the contour on the original image
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)

                # Calculate the centroid using moments
                M = cv2.moments(contour)
                if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if (radius > 0):
                            Z = (f_x * real_radius) / radius
                            X = (cx * Z) / f_x
                            Y = (cy * Z) / f_y

                            cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        return img
    
    def findBlueAndTan(self, img, blackwhite):

        K = np.array([[2564.3186869, 0, 0], 
                    [0, 2569.70273111, 0], 
                    [0, 0, 1]])

        # Focal length in pixels (from the intrinsic matrix)
        f_x = K[0, 0]
        f_y = K[1, 1]
        real_radius = 10

        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(blackwhite, cv2.COLOR_BGR2HSV)

        # Define the lower and upper range for blue color in HSV
        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Define the lower and upper range for tan color in HSV
        lower_tan = np.array([10, 0, 0])
        upper_tan = np.array([30, 255, 255])

        # Create masks for blue and tan colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_tan = cv2.inRange(hsv, lower_tan, upper_tan)

        # Perform morphological operations to remove noise from the masks
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        mask_tan = cv2.morphologyEx(mask_tan, cv2.MORPH_CLOSE, kernel)
        mask_tan = cv2.morphologyEx(mask_tan, cv2.MORPH_OPEN, kernel)

        # Combine blue and tan masks
        mask = mask_blue + mask_tan

        # Find contours for combined mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours and calculate depth at the centroid
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) != 8:
                    cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
                    M = cv2.moments(contour)

                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        if (radius > 0):
                            Z = (f_x * real_radius) / radius
                            X = (cx * Z) / f_x
                            Y = (cy * Z) / f_y

                            cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img
    
    def findBlueAndTanIn3D(self, img, blackwhite):

        K = np.array([[2564.3186869, 0, 0], 
                    [0, 2569.70273111, 0], 
                    [0, 0, 1]])

        # Focal lengths
        f_x = K[0, 0]
        f_y = K[1, 1]

        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(blackwhite, cv2.COLOR_BGR2HSV)

        # Define the lower and upper range for blue color in HSV
        lower_blue = np.array([80, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Define the lower and upper range for tan color in HSV
        lower_tan = np.array([10, 0, 0])
        upper_tan = np.array([30, 255, 255])

        # Create masks for blue and tan colors
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        mask_tan = cv2.inRange(hsv, lower_tan, upper_tan)

        # Perform morphological operations to remove noise from the masks
        kernel = np.ones((5, 5), np.uint8)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)
        mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)

        mask_tan = cv2.morphologyEx(mask_tan, cv2.MORPH_CLOSE, kernel)
        mask_tan = cv2.morphologyEx(mask_tan, cv2.MORPH_OPEN, kernel)

        # Combine blue and tan masks
        mask = mask_blue + mask_tan

        # Find contours for the combined mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        real_radius = 10  # Circle's radius in inches

        # Process each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Filter based on area
                # Fit a minimum enclosing circle to the contour
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)

                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                
                cv2.drawContours(img, [approx], 0, (0, 0, 255), 2) 
                M = cv2.moments(contour)

                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if (radius > 0):
                        Z = (f_x * real_radius) / radius
                        X = (cx * Z) / f_x
                        Y = (cy * Z) / f_y

                        cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


                # Calculate the depth (Z) using the formula: Z = f_x * real_radius / image_radius
                if radius > 0:
                    Z = (f_x * real_radius) / radius

                    # Calculate real-world X and Y using the pixel coordinates (x, y)
                    X = (x * Z) / f_x
                    Y = (y * Z) / f_y

                    # Mark the circle and display the 3D coordinates
                    cv2.circle(img, center, 1, (0, 255, 0), 2)
                    cv2.putText(img, f"X: {X:.2f}, Y: {Y:.2f}, Z: {Z:.2f}", (center[0] - 50, center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img