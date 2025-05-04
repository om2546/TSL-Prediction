import cv2
import threading
import time
import numpy as np
import mediapipe as mp
import keras
from keras.models import load_model
import math
from scipy.spatial import distance


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

model = load_model('gru_2layer_seed0.keras')
START = time.time()
#Global variable
pred_result = 'noaction'
running = True
pred_time = 0
frame_queue = []
prob = ""

words_list = ['1-hello',
              '2-ty',
              '3-sorry',
              '4-yes',
              '5-no',
              '6-nopb',
              '7-fun',
              '8-you',
              '9-i',
              '10-sports',
              '11-tuesday',
              '12-fine',
              '13-sick',
              '14-study',
              '15-signlang',
              '16-what',
              '17-like',
              '18-howmuch',
              '19-football',
              '20-fruit',
              '21-watermelon',
              '22-eat',
              '23-hungry',
              '24-sickcold',
              '25-noodle',
              '26-tomorrow',
              '27-headache',
              '28-age',
              '29-thirty',
              '30-name',
              '31-noaction',]

def extract_landmarks(results,
                      face_indices = [[0, 2], [2, 7], [0, 9], [0, 5], [5, 8], [0, 10]],
                      right_body_indices = [[11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [0, 15]],
                      left_body_indices = [[12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [0, 16]],
                      ):
    # Function to extract and concatenate all landmarks into a single vector
    # Default indices are for all landmarks
    
    landmarks = []
    
    # Body Pose Landmarks
    if results.pose_landmarks:
        
        #center point
        center = [(results.pose_landmarks.landmark[11].x + results.pose_landmarks.landmark[12].x) / 2.0,
                  (results.pose_landmarks.landmark[11].y + results.pose_landmarks.landmark[12].y) / 2.0]
        normdist = distance.euclidean(center,
                                      [results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
        # landmarks.extend(center)
        
        # 18 angles + 18 distances for body & face
        for indices_list in [face_indices, right_body_indices, left_body_indices]:
            for pair in indices_list:
                landmarks.extend([
                                math.atan2( results.pose_landmarks.landmark[pair[1]].y - results.pose_landmarks.landmark[pair[0]].y,
                                            results.pose_landmarks.landmark[pair[1]].x - results.pose_landmarks.landmark[pair[0]].x),
                                
                                distance.euclidean( [results.pose_landmarks.landmark[pair[0]].x, results.pose_landmarks.landmark[pair[0]].y],
                                                    [results.pose_landmarks.landmark[pair[1]].x, results.pose_landmarks.landmark[pair[1]].y])/normdist
                                ])
        
        # 2 angles from center
        landmarks.extend([
                        math.atan2( center[1] - results.pose_landmarks.landmark[11].y,
                                    center[0] - results.pose_landmarks.landmark[11].x),
                        
                        math.atan2( center[1] - results.pose_landmarks.landmark[12].y,
                                    center[0] - results.pose_landmarks.landmark[12].x)
                        ])
        
        # Left Hand Landmarks
        if results.left_hand_landmarks:
            for finger in range(5):
                landmarks.extend([
                                math.atan2( results.left_hand_landmarks.landmark[1+finger*4].y - results.left_hand_landmarks.landmark[0].y,
                                            results.left_hand_landmarks.landmark[1+finger*4].x - results.left_hand_landmarks.landmark[0].x),
                                
                                distance.euclidean( [results.left_hand_landmarks.landmark[0].x, results.left_hand_landmarks.landmark[0].y],
                                                    [results.left_hand_landmarks.landmark[1+finger*4].x, results.left_hand_landmarks.landmark[1+finger*4].y])/normdist])
                
                for joint in [2,3,4]:
                    landmarks.extend([
                                        math.atan2( results.left_hand_landmarks.landmark[joint+finger*4].y - results.left_hand_landmarks.landmark[joint+finger*4-1].y,
                                                    results.left_hand_landmarks.landmark[joint+finger*4].x - results.left_hand_landmarks.landmark[joint+finger*4-1].x),
                                        
                                        distance.euclidean( [results.left_hand_landmarks.landmark[joint+finger*4-1].x, results.left_hand_landmarks.landmark[joint+finger*4-1].y],
                                                            [results.left_hand_landmarks.landmark[joint+finger*4].x, results.left_hand_landmarks.landmark[joint+finger*4].y])/normdist])
                
            for fingertips in range(1,5):
                landmarks.extend([
                                math.atan2( results.left_hand_landmarks.landmark[fingertips*4].y - results.left_hand_landmarks.landmark[(fingertips+1)*4].y,
                                            results.left_hand_landmarks.landmark[fingertips*4].x - results.left_hand_landmarks.landmark[(fingertips+1)*4].x),
                                
                                distance.euclidean( [results.left_hand_landmarks.landmark[(fingertips+1)*4].x, results.left_hand_landmarks.landmark[(fingertips+1)*4].y],
                                                    [results.left_hand_landmarks.landmark[fingertips*4].x, results.left_hand_landmarks.landmark[fingertips*4].y])/normdist])
        else:  
            landmarks.extend([0] * 48) # Placeholder for 24 hand landmarks * 2 attributes            
        
        # Right Hand Landmarks
        if results.right_hand_landmarks:
            for finger in range(5):
                landmarks.extend([
                                math.atan2( results.right_hand_landmarks.landmark[1+finger*4].y - results.right_hand_landmarks.landmark[0].y,
                                            results.right_hand_landmarks.landmark[1+finger*4].x - results.right_hand_landmarks.landmark[0].x),
                                
                                distance.euclidean( [results.right_hand_landmarks.landmark[0].x, results.right_hand_landmarks.landmark[0].y],
                                                    [results.right_hand_landmarks.landmark[1+finger*4].x, results.right_hand_landmarks.landmark[1+finger*4].y])/normdist])
                
                for joint in [2,3,4]:
                    landmarks.extend([
                                        math.atan2( results.right_hand_landmarks.landmark[joint+finger*4].y - results.right_hand_landmarks.landmark[joint+finger*4-1].y,
                                                    results.right_hand_landmarks.landmark[joint+finger*4].x - results.right_hand_landmarks.landmark[joint+finger*4-1].x),
                                        
                                        distance.euclidean( [results.right_hand_landmarks.landmark[joint+finger*4-1].x, results.right_hand_landmarks.landmark[joint+finger*4-1].y],
                                                            [results.right_hand_landmarks.landmark[joint+finger*4].x, results.right_hand_landmarks.landmark[joint+finger*4].y])/normdist])
            for fingertips in range(1,5):
                landmarks.extend([
                                math.atan2( results.right_hand_landmarks.landmark[fingertips*4].y - results.right_hand_landmarks.landmark[(fingertips+1)*4].y,
                                            results.right_hand_landmarks.landmark[fingertips*4].x - results.right_hand_landmarks.landmark[(fingertips+1)*4].x),
                                
                                distance.euclidean( [results.right_hand_landmarks.landmark[(fingertips+1)*4].x, results.right_hand_landmarks.landmark[(fingertips+1)*4].y],
                                                    [results.right_hand_landmarks.landmark[fingertips*4].x, results.right_hand_landmarks.landmark[fingertips*4].y])/normdist])
        else:  
            landmarks.extend([0] * 48) # Placeholder for 24 hand landmarks * 2 attributes   
        
        
    else:
        landmarks.extend([0] * 134) # Placeholder for all 134 attributes


    return np.array(landmarks)

def activate_webcam():
    global running
    global pred_result
    print("Activating webcam...")
    cap = cv2.VideoCapture(0)   
    print("Webcam activated")
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while running and cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            start_time = time.time()
            # Process image
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            landmarks = extract_landmarks(results)
            landmarks_time = time.time()
            landmarks = np.append(landmarks, landmarks_time)
            frame_queue.append(landmarks)
            # print("Queue size from webcam: ", len(frame_queue))
            
            # # Append landmarks to log.csv
            # with open('log.csv', 'a') as f:
            #     np.savetxt(f, [landmarks], delimiter=',', fmt='%.3f')
            
            end_time = time.time()

            # Display data
            image = cv2.flip(image, 1)
            cv2.putText(image, "Extract time: {:.2f}ms".format((end_time - start_time) * 1000), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "FPS: {:.2f}".format(1 / (end_time - start_time)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Prediction: {}".format(pred_result), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Prediction time: {:.2f}ms".format(pred_time * 1000), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, "Prob: "+prob, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('MediaPipe Holistic - Selected Landmarks', image)

            # print("Queue size: ", frame_queue.qsize())
            
            if cv2.waitKey(5) & 0xFF == 27:
                running = False
                break

    cap.release()
    cv2.destroyAllWindows()

def model_predict(value):
    global prob
    pred = model.predict(value,verbose=0)
    # max value in pred
    prob = words_list[np.argmax(pred)] + " : " + f"{np.max(pred):.2f}"
    if(np.max(pred) > 0.75):
        index_max = np.argmax(pred)
        return words_list[index_max]
    else:
        return "Unknown"

def interpolate_data(data,frame=75,sec=3):
    # Extract time stamp data
    time_stamp = data[:,-1]
    first_time = time_stamp[0]
    time_stamp = time_stamp - first_time
    # Extract data without time stamp
    data = data[:,:-1]
    # Create new time stamp data
    new_time_stamp = np.linspace(0,sec,frame)
    # Create new data
    new_data = np.zeros((frame,data.shape[1]))
    for i in range(data.shape[1]):
        new_data[:,i] = np.interp(new_time_stamp,time_stamp,data[:,i])
    return new_data

def get_last_n_sec_data(data,sec=3):
    last_timestamp = data[-1, -1]  # Get the timestamp of the last row
    for i in range(len(data) - 1, -1, -1):  # Iterate from the last row backward
        if last_timestamp - data[i, -1] >= sec:  # Check if the time difference is 3 seconds
            return data[i:, :]  # Return rows from this index to the last row

    return data  # If no row meets the condition, return all data

def predict():
    global running
    global pred_result
    global pred_time
    global frame_queue
    global prob
    while running:
        # print("Predicting... queue size: ", len(frame_queue))
        if len(frame_queue) >= 1 and (frame_queue[-1][-1] - frame_queue[0][-1]) >= 3:
            input_data = np.array(frame_queue)
            # Get Last 3 seconds data
            np.savetxt("input.csv", input_data, delimiter=",")      
            with open('log.txt', 'a') as f:
                f.write("data_shape: " + str(input_data.shape) + "\n")
            
            input_data = get_last_n_sec_data(input_data)
            test_data = np.empty(shape=(1, 75, 134))
            test_data[0] = interpolate_data(input_data)

            # Cut 10 frames from the beginning if frame queue is more than 100
            if len(frame_queue) > 100:
                frame_queue = frame_queue[10:]
                
            
            # # save input data as csv
            # np.savetxt("input.csv", input_data, delimiter=",")
            # print("SAVE")
            # np.savetxt("input_data.csv", test_data[0], delimiter=",")
            
            
            start_time = time.time()
            pred_result = model_predict(test_data)
            end_time = time.time()
            pred_time = end_time - start_time
            # print("Time:",end_time - START)
            # print("First Frame Time:",input_data[0][-1]-START)
            # print("Last Frame Time:",input_data[-1][-1]-START)
            # print("Predicted: ", pred_result)
            
if __name__ == "__main__":
    # Create threads
    webcam_thread = threading.Thread(target=activate_webcam)
    predict_thread = threading.Thread(target=predict)

    # Start threads
    webcam_thread.start()
    predict_thread.start()

    # Wait for both threads to finish
    webcam_thread.join()
    predict_thread.join()

