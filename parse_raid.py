#!/usr/bin/python3
import cv2
import numpy as np
from sys import argv
from PIL import Image
from tesserocr import PyTessBaseAPI

def getWhiteArea(hsv, low, high):
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    return mask

def main():
    print('Starting')
    if len(argv) < 2:
        print("Usage: script file.jpg")
        exit()

    img_file = argv[1]
    img = cv2.imread(img_file)

    print("Read image")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gym_area  = hsv[70:140, 140:700]
    time_area = hsv[767:800, 540:670]
    cp_area   = hsv[205:310, 217:580]
    poke_area = hsv[300:400, 20:700]        

    gym_m = getWhiteArea(gym_area,   np.array([0,0,210]), np.array([180, 80, 255]))
    time_m = getWhiteArea(time_area, np.array([0,0,230]), np.array([180, 60, 255]))
    cp_m = getWhiteArea(cp_area,     np.array([0,0,252]), np.array([180, 20, 255]))
    poke_m = getWhiteArea(poke_area, np.array([0,0,250]), np.array([180, 25, 255]))

    # convert to Pillow format
#    gym_m = cv2.cvtColor(gym_m, cv2.COLOR_GRAY2RGB)
    gym_pm = Image.fromarray(gym_m)

#    time_m = cv2.cvtColor(time_m, cv2.COLOR_BGR2RGB)
    time_pm = Image.fromarray(time_m)

#    cp_m = cv2.cvtColor(cp_m, cv2.COLOR_BGR2RGB)    
    cp_pm = Image.fromarray(cp_m)
    
#    poke_m = cv2.cvtColor(poke_m, cv2.COLOR_BGR2RGB)
    poke_pm = Image.fromarray(poke_m)

    gym_text_eng = ""

    with PyTessBaseAPI() as api:
        
        api.SetImage(gym_pm)
        gym_text_eng = api.GetUTF8Text()
        
        # for time
        api.SetVariable('tessedit_char_whitelist', '0123456789:')
        api.SetImage(time_pm)
    
        print("Raid Time: ", api.GetUTF8Text())
    
        api.SetVariable('tessedit_char_whitelist', '0123456789')
        api.SetImage(cp_pm)
        print("Boss CP: ", api.GetUTF8Text())
    
        api.SetVariable('tessedit_char_whitelist', 'abcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ')
        api.SetImage(poke_pm)
        print("Poke Name: ", api.GetUTF8Text())


    with PyTessBaseAPI(lang='rus') as api:
        api.SetImage(gym_pm)
        print("Gym RUS: ", api.GetUTF8Text())
        print("Gym ENG: ", gym_text_eng)

main()

