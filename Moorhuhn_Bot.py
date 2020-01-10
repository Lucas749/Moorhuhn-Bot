# Import modules
import os
import pynput
import pandas as pd
from PIL import ImageGrab
import time
import numpy as np
import scipy.spatial.distance as dist
from collections import Counter
from sklearn.cluster import KMeans
import pyautogui as gui
from sklearn.preprocessing import StandardScaler
import joblib
import cv2


#############################################################
# Functions
#############################################################
# Creates a screenshot from the game
def screenshot(position_parameters):
    x1 = position_parameters.loc[0, 'X1_Pos']
    y1 = position_parameters.loc[0, 'Y1_Pos']
    x2 = position_parameters.loc[0, 'X2_Pos']
    y2 = position_parameters.loc[0, 'Y2_Pos']
    image = np.array(ImageGrab.grab(bbox=[x1, y1, x2, y2]))
    return image


# Score recognition
def score_recognition(position_parameters, reco_model, digit_scaler):
    position_parameters["X1_Pos"] = position_parameters["X2_Pos"] - 160
    position_parameters["Y2_Pos"] = position_parameters["Y1_Pos"] + 90
    position_parameters["Y1_Pos"] = position_parameters["Y1_Pos"] + 10
    x1 = position_parameters.loc[0, 'X1_Pos']
    y1 = position_parameters.loc[0, 'Y1_Pos']
    x2 = position_parameters.loc[0, 'X2_Pos']
    y2 = position_parameters.loc[0, 'Y2_Pos']
    im = np.array(ImageGrab.grab(bbox=[x1, y1, x2, y2]))

    # Convert to grayscale
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Create thresholds
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    cont = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Create rectangles for contours
    rects = [cv2.boundingRect(c) for c in cont]
    im_gray = cv2.bitwise_not(im_gray)
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)
    im_th = cv2.bitwise_not(im_th)

    score_list = []
    # Check if rects do not accidentally detect more numbers
    if len(rects) <= 4:
        for i, rect in enumerate(rects):
            number = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            number = cv2.resize(number, (28, 28), interpolation=cv2.INTER_AREA)
            number = cv2.dilate(number, (3, 3))
            number = number.flatten().reshape(1, -1)
            number = digit_scaler.transform(number)
            prediction = reco_model.predict(number)[0]
            score_list.append(prediction)
        score_list = score_list[::-1]
        score = int(''.join(map(str, score_list)))
        return score
    return "Error"


# On_click function for listeners
def on_click(x, y, button, pressed):
    global number_clicks
    global positions
    if pressed:
        number_clicks = number_clicks + 1
        positions.append([x, y])
        if (number_clicks == 2):
            return False


# On_release function for listener
def on_release(key):
    global esc_press
    if key == pynput.keyboard.Key.esc:
        esc_press = False
        return False


# Returns position of the game
def game_position():
    global number_clicks
    global positions
    number_clicks = 0
    positions = []
    print("Please first click on the top left corner of the game and then on the bottom right")
    listener = pynput.mouse.Listener(on_click=on_click)
    listener.start()
    while listener.running == True:
        if len(positions) == 2:
            break
    game_position = pd.DataFrame({'X1_Pos': [positions[0][0]], 'Y1_Pos': [positions[0][1]], 'X2_Pos': [positions[1][0]],
                                  'Y2_Pos': [positions[1][1] - 175]})
    return game_position


#############################################################
# Program logic
#############################################################
# Define variables and constants
SMALL_WINDOW = 150
DISTANCE_THRESHOLD = 150
GAME_LENGTH = 90
PIXEL_THRESHOLD_UPPER = 1
PIXEL_THRESHOLD_LOWER = 5
UPPER_DOWNSHIFT = 0
LOWER_DOWNSHIFT = 0
SLEEP_TIME = 0
LOWER_CLUSTER = 5
UPPER_CLUSTER = 4
FILTER_LARGEST = False
esc_press = True

if __name__ == "__main__":
    # Retrieve game position
    position_parameters = game_position()
    time.sleep(3)

    # Set game end time
    end_time = time.time() + GAME_LENGTH

    # Set up mouse and keyboard control
    mouse = pynput.mouse.Controller()
    keyboard = pynput.keyboard.Controller()
    shots_fired = 0
    MOUSE_REST = ((position_parameters.loc[0, 'X1_Pos'] + position_parameters.loc[0, 'X2_Pos']) / 2,
                  position_parameters.loc[0, 'Y1_Pos'] + 30)

    # Start listener to quick exit bot
    listener = pynput.keyboard.Listener(on_release=on_release)
    listener.start()
    print("\n Press esc to kill bot before time is up")

    # Load model for score recognition
    os.chdir("MODEL_SAVE_PATH")
    reco_model = joblib.load("NN_number_recognition_balanced")
    digit_scaler = joblib.load("digit_scaler")

    # Start the game
    mouse.position = ((position_parameters.loc[0, 'X1_Pos'] + 170), position_parameters.loc[0, 'Y2_Pos'] + 80)
    mouse.click(pynput.mouse.Button.left)

    # Start shooting routine
    while time.time() < end_time and esc_press:
        mouse.position = MOUSE_REST

        # Take screenshot
        screen = screenshot(position_parameters)
        pixel_values = screen

        # Separate screenshot in two parts to better determine smaller birds (otherwise KMeans would be biased towards bigger birds)
        pixel_values_upper = pixel_values[:SMALL_WINDOW, :, :]
        pixel_values_lower = pixel_values[SMALL_WINDOW:, :, :]

        # Search for red pixels in upper screen
        red = np.multiply(pixel_values_upper[:, :, 0] >= [190], pixel_values_upper[:, :, 0] <= [255])
        blue = pixel_values_upper[:, :, 1] <= [10]
        green = pixel_values_upper[:, :, 2] <= [10]

        targets_upper = np.multiply(red, blue, green)
        coordinates_upper = np.where(targets_upper == 1)
        coordinates_upper = pd.DataFrame({"X": coordinates_upper[1], "Y": coordinates_upper[0]})

        # Search for red pixels in lower screen
        red = np.multiply(pixel_values_lower[:, :, 0] >= [190], pixel_values_lower[:, :, 0] <= [255])  # 220
        blue = pixel_values_lower[:, :, 1] <= [10]
        green = pixel_values_lower[:, :, 2] <= [10]

        targets_lower = np.multiply(red, blue, green)
        coordinates_lower = np.where(targets_lower == 1)
        coordinates_lower = pd.DataFrame({"X": coordinates_lower[1], "Y": coordinates_lower[0] + SMALL_WINDOW})

        # Initiate KMeans
        model_lower = KMeans(n_clusters=LOWER_CLUSTER, random_state=45)
        model_upper = KMeans(n_clusters=UPPER_CLUSTER, random_state=45)
        try:
            # Apply different filter to KMeans outputs
            if coordinates_lower.shape[0] > 0:
                model_lower.fit(coordinates_lower)
                label_count_lower = pd.DataFrame(np.unique(model_lower.labels_, return_counts=True))
                pixel_check = np.where((label_count_lower.iloc[1, :] > PIXEL_THRESHOLD_LOWER) == True)[0]
                label_count_lower = label_count_lower[pixel_check]
                center_lower = model_lower.cluster_centers_[label_count_lower.columns]
                center_lower[:, 1] = center_lower[:, 1] + LOWER_DOWNSHIFT

                if FILTER_LARGEST:
                    center_index_lower = label_count_lower.iloc[1, :].nlargest(
                        int(label_count_lower.shape[1] / 2)).index
                    center_lower = model_lower.cluster_centers_[center_index_lower]
                centers = center_lower

            if coordinates_upper.shape[0] > 0:
                model_upper.fit(coordinates_upper)
                label_count_upper = pd.DataFrame(np.unique(model_upper.labels_, return_counts=True))
                pixel_check = np.where((label_count_upper.iloc[1, :] > PIXEL_THRESHOLD_UPPER) == True)[0]
                label_count_upper = label_count_upper[pixel_check]
                center_upper = model_upper.cluster_centers_[label_count_upper.columns]
                center_upper[:, 1] = center_upper[:, 1] + UPPER_DOWNSHIFT

                if FILTER_LARGEST:
                    center_index_upper = label_count_upper.iloc[1, :].nlargest(
                        int(label_count_upper.shape[1] / 2)).index
                    center_upper = model_upper.cluster_centers_[center_index_upper]
                centers = center_upper

            if coordinates_upper.shape[0] > 0 and coordinates_lower.shape[0] > 0:
                centers = np.concatenate((center_upper, center_lower), axis=0)

            np.random.shuffle(centers)

            # Filter out too close clusters
            for i in range(0, 5):
                distance = dist.cdist(centers, centers)
                criteria = np.multiply(distance < DISTANCE_THRESHOLD, (distance != 0))
                drop_centers = np.array([np.array(np.where(criteria == 1))[0][:int(np.sum(criteria == 1) / 2)],
                                         np.array(np.where(criteria == 1))[1][:int(np.sum(criteria == 1) / 2)]])
                use_centers = np.array(list(range(0, centers.shape[0])) + list(np.unique(drop_centers)[:]))
                use_centers = [k for k, v in Counter(list(use_centers)).items() if v == 1]
                use_centers = use_centers + list(np.unique(drop_centers[0, :]))
                centers = centers[use_centers, :]
                np.random.shuffle(centers)

            # Shoot on every identified target
            for i in range(0, centers.shape[0]):
                mouse.position = (position_parameters.loc[0, 'X1_Pos'] + centers[i, 0],
                                  position_parameters.loc[0, 'Y1_Pos'] + centers[i, 1])
                mouse.click(pynput.mouse.Button.left)
                mouse.position = MOUSE_REST
                shots_fired = shots_fired + 1

                # Reload if ammunition is empty
                if (shots_fired == 10):
                    shots_fired = 0
                    keyboard.press(pynput.keyboard.Key.space)
                    keyboard.release(pynput.keyboard.Key.space)

            score = score_recognition(position_parameters.copy(), reco_model, digit_scaler)
            if type(score) == int:
                score_output = score
            time.sleep(SLEEP_TIME)
        except:
            time.sleep(SLEEP_TIME)

    listener.stop()
    print("Score:", score_output)
