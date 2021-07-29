import cv2
import numpy as np
import glob
import random
import base64

# Name custom object
classes = ["pile"]

def predict_humanView(file,allResult):
    # Load Yolo
        humanVeiw = cv2.dnn.readNet("./model/Human_Veiw.weights", "./model/custom-yolov4-detector.cfg")

    #import images 
        images_path = glob.glob(f"./write_humanVeiw_Img/{file}")
        layer_names = humanVeiw.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in humanVeiw.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # Insert here the path of your images
        # loop through all the images
        for img_path in images_path:
                # print(img_path)
                # Loading image
                img = cv2.imread(img_path)
                # img = cv2.resize(img, None, fx=0.3, fy=0.3)
                img = cv2.resize(img, (416,416))
                height, width, channels = img.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                humanVeiw.setInput(blob)
                outs = humanVeiw.forward(output_layers)
                # Showing informations on the screen
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                    
                        if confidence > 0.5:
                        
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
    
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                # print(confidences)
                # print(indexes)
                font = cv2.FONT_HERSHEY_PLAIN
                count = 0
                
                for i in range(len(boxes)):
                    
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        count+=1
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
                        
                        cv2.putText(img, label+ str(round(confidences[i],1)), (x, y-20 ), font, 1, (0,0,255), 2)
                        # cv2.putText(img, str(round(confidences[i],2)), (x, y + 30), font, 3, color, 2)
                        
                cv2.putText(img, "count : "+str(count), (0, 30), font, 2, (255,255,0), 2)

                retval, buffer = cv2.imencode('.jpg', img)
                baseImg = base64.b64encode(buffer)
                allResult.append({"img":baseImg,"count":str(count)})
                # baseImg = base64.b64encode(img)
                # cv2.imshow("Image", img)
               
                # key = cv2.waitKey(0)

                # cv2.destroyAllWindows()
                # cv2.imwrite('./test_accu_model_3/'+img_path.replace("./24_6/test/test ",""),img)
                # print(len(allResult))
        return allResult
           
def predict_droneVeiw(file,allResult):
    # Load Yolo
        humanVeiw = cv2.dnn.readNet("./model/Drone_Veiw.weights", "./model/custom-yolov4-detector.cfg")
    #import images 

        images_path = glob.glob(f"./write_droneVeiw_Img/{file}")
        layer_names = humanVeiw.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in humanVeiw.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        # Insert here the path of your images
        # loop through all the images
        for img_path in images_path:
                # print(img_path)
                # Loading image
                img = cv2.imread(img_path)
                # img = cv2.resize(img, None, fx=0.3, fy=0.3)
                img = cv2.resize(img, (416,416))
                height, width, channels = img.shape

                # Detecting objects
                blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                humanVeiw.setInput(blob)
                outs = humanVeiw.forward(output_layers)
                # Showing informations on the screen
                class_ids = []
                confidences = []
                boxes = []
                
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                    
                        if confidence > 0.5:
                        
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)

                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
    
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                # print(confidences)
                # print(indexes)
                font = cv2.FONT_HERSHEY_PLAIN
                count = 0
                
                for i in range(len(boxes)):
                    
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        count+=1
                        label = str(classes[class_ids[i]])
                        color = colors[class_ids[i]]
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
                        
                        cv2.putText(img, label+ str(round(confidences[i],1)), (x, y-20 ), font, 1, (0,0,255), 2)
                        # cv2.putText(img, str(round(confidences[i],2)), (x, y + 30), font, 3, color, 2)
                        
                cv2.putText(img, "count : "+str(count), (0, 30), font, 2, (255,255,0), 2)

                retval, buffer = cv2.imencode('.jpg', img)
                baseImg = base64.b64encode(buffer)
                allResult.append({"img":baseImg,"count":str(count)})
                # baseImg = base64.b64encode(img)
                # cv2.imshow("Image", img)
               
                # key = cv2.waitKey(0)

                # cv2.destroyAllWindows()
                # cv2.imwrite('./test_accu_model_3/'+img_path.replace("./24_6/test/test ",""),img)
                # print(len(allResult))
        return allResult