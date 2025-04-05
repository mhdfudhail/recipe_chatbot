import cv2
import time
import os
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import cv2.aruco as aruco
import serial_collect


# Configuration
# MODEL_PATH = r'C:\Users\models\best_908.pt'
MODEL_PATH = r"C:\Users\models\best_908.pt"
CONFIDENCE = 0.45

def load_model(model_path):
    """Load the YOLO model."""
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path, False)
    return model

def infer_image(model, image):
    """Run inference on an image and return result with visualization."""
    if model is None:
        return None, None
    
    start_time = time.time()
    results = model.predict(source=image, conf=CONFIDENCE, save=False)
    inference_time = time.time() - start_time
    
    # Get the visualization
    for r in results:
        im_array = r.plot()
        annotated_img = Image.fromarray(im_array[..., ::-1])  # Convert BGR to RGB
    
    return annotated_img, results, inference_time

def findArucoMarker(img, markerSize=6, totalMarker=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    arucoParam = aruco.DetectorParameters()
    bbx, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)

    if draw:
        aruco.drawDetectedMarkers(img, bbx)

    return [bbx, ids]

def display_results(results, inference_time):
    detected_list=[]
    """Display detection results."""
    if results is None or len(results) == 0:
        print("No detections made.")
        return
    
    # Count detected objects by class
    counter = Counter(results[0].boxes.cls.cpu().numpy().astype(int))
    
    print(f"\nDetection Results (inference time: {inference_time:.2f} seconds):")
    print("-" * 50)
    
    # Display counts for each detected class
    for class_id, count in counter.items():
        class_name = results[0].names[class_id]
        detected_list.append((f"{count} {class_name}"))
        print(f"{count} {class_name}")
    
    print("-" * 50)
    print(detected_list)
    return detected_list

def process_image_file(model, image_path):
    veg_list = []
    """Process an image from a file."""
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found at {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    
    # Load the image
    try:
        image = Image.open(image_path)
        annotated_img, results, inference_time = infer_image(model, image)
        
        if annotated_img:
            # Save and show results
            save_path = f"output_{os.path.basename(image_path)}"
            annotated_img.save(save_path)
            print(f"Annotated image saved to {save_path}")
            
            # Display results
            veg_list = display_results(results, inference_time)
            
            # Show the image (optional)
            annotated_img.show()
    except Exception as e:
        print(f"Error processing image: {e}")
    return veg_list

def capture_from_camera(model):
    """Capture a single frame from camera and process it."""
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press SPACE to capture an image or ESC to exit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Display live preview
        cv2.imshow('Camera Preview (Press SPACE to capture, ESC to exit)', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key - exit
        if key == 27:
            break
        # SPACE key - capture and process
        elif key == 32:
            print("Image captured, processing...")
            
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process the captured frame
            annotated_img, results, inference_time = infer_image(model, pil_image)
            
            if annotated_img:
                # Save the result
                timestamp = int(time.time())
                save_path = f"camera_capture_{timestamp}.jpg"
                annotated_img.save(save_path)
                print(f"Annotated image saved to {save_path}")
                
                # Display results
                display_results(results, inference_time)
                
                # Show the processed image
                cv2.imshow('Captured Result', np.array(annotated_img)[:, :, ::-1])
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def image_from_camera(model):
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ret, frame = cap.read()
    if ret:
        print("image captured!")
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process the captured frame
        annotated_img, results, inference_time = infer_image(model, pil_image)
        
        if annotated_img:
            # Save the result
            timestamp = int(time.time())
            save_path = f"camera_capture_{timestamp}.jpg"
            annotated_img.save(save_path)
            print(f"Annotated image saved to {save_path}")
            
            # Display results
            veg_list = display_results(results, inference_time)
            print(veg_list)

    else:
        print("Error: Failed to capture image")
    
    cap.release()
    cv2.destroyAllWindows()
    return veg_list

def flour_aruco():
    markerDict = {}
    aruco_list=[]
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ret, frame = cap.read()
    if ret:
        print("image captured!")
        arucoFound = findArucoMarker(frame)
        if len(arucoFound[0])!=0:
            for bbx, ids in zip(arucoFound[0], arucoFound[1]):
                idCord = [bbx]
                # print(f"full cord: {idCord}")
                # print("fist:",bbx[0][0])
                # print("second:",bbx[0][1])
                idNum = ids[0]
                idCord = [int(bbx[0][0][0]),
                            int(bbx[0][0][1]),
                            int(bbx[0][1][0]),
                            int(bbx[0][1][1]),
                            int(bbx[0][2][0]),
                            int(bbx[0][2][1])]
                # print(idCord)
                marker={idNum:idCord}
                markerDict.update(marker)
        print(markerDict)
        if 10 in markerDict.keys():
            aruco_list.append("flour_1")
        if 20 in markerDict.keys():
            aruco_list.append("flour_2")


    else:
        print("Error: Failed to capture image")
    
    cap.release()
    cv2.destroyAllWindows()
    return aruco_list

def veg_flour(model):
    markerDict = {}
    aruco_list=[]
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    time.sleep(1)
    ret, frame = cap.read()
    if ret:
        print("image captured!")
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Process the captured frame
        annotated_img, results, inference_time = infer_image(model, pil_image)
        
        if annotated_img:
            # Save the result
            timestamp = int(time.time())
            save_path = f"camera_capture_{timestamp}.jpg"
            annotated_img.save(save_path)
            print(f"Annotated image saved to {save_path}")
            
            # Display results
            veg_list = display_results(results, inference_time)
            print(veg_list)

        arucoFound = findArucoMarker(frame)
        if len(arucoFound[0])!=0:
            for bbx, ids in zip(arucoFound[0], arucoFound[1]):
                idCord = [bbx]
                # print(f"full cord: {idCord}")
                # print("fist:",bbx[0][0])
                # print("second:",bbx[0][1])
                idNum = ids[0]
                idCord = [
                        int(bbx[0][0][0]),
                        int(bbx[0][0][1]),
                        int(bbx[0][1][0]),
                        int(bbx[0][1][1]),
                        int(bbx[0][2][0]),
                        int(bbx[0][2][1])]
                # print(idCord)
                marker={idNum:idCord}
                markerDict.update(marker)
        print(markerDict)
        if 10 in markerDict.keys():
            aruco_list.append("flour_1")
        if 20 in markerDict.keys():
            aruco_list.append("flour_2")    

    else:
        print("Error: Failed to capture image")
    
    cap.release()
    cv2.destroyAllWindows()
    return veg_list, aruco_list

def get_weight():
    weight_list = serial_collect.collectData()
    return weight_list

def check_shelf(model):
    # weights = get_weight()
    weights = 0
    veg, ingr= veg_flour(model)
    print(f"vegs:{veg}, ingredients:{ingr}, corresponding weights:{weights}")
    return {'vegs':[veg],'ingredients':[ingr], 'weights':weights}
    # return(f"vegitables:{veg}, ingredients:{ingr}, corresponding weights:{weights}")


def main():
    # Load the model
    model = load_model(MODEL_PATH)
    if model is None:
        return
    
    while True:
        print("\nYOLO Object Detection")
        print("1. Process image from file")
        print("2. Capture from camera")
        print("3. Exit")
        print("4. aruco detection")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            image_path = input("Enter the path to the image file: ")
            process_image_file(model, image_path)
        elif choice == '2':
            print(image_from_camera(model))
        elif choice == '4':
            # items_and_weight ={}
            # weights = get_weight()
            # veg, flour= veg_flour(model)
            # # items_and_weight={veg[0]:weights[0],
            # #                   veg[1]:weights[1],
            # #                   flour[0]:weights[2],
            # #                   flour[1]:weights[3]}
            # print(f"vegs:{veg}, flour:{flour}, corresponding weights:{weights}")
            print(check_shelf(model))


            # flour_aruco()
            # print()
            # break
        elif choice == '3':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()