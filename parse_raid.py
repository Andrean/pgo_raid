#!/usr/bin/python3
import cv2
import numpy as np
import config
from sys import argv
from PIL import Image
from tesserocr import PyTessBaseAPI


# input has relative values
class Area(object):
    def __init__(self, coords, img, relative=False, relativeCenterX=False):
        self.img = img
        rows, cols, ch = img.shape

        if relative:
            coords = list(map(lambda x: int(x*rows), coords))
            if relativeCenterX:
                coords[2] += int(cols/2)		
                coords[3] += int(cols/2)
        self.y0 = coords[0]
        self.y1 = coords[1]
        self.x0 = coords[2]
        self.x1 = coords[3]		
        self.roi = img[self.y0:self.y1, self.x0:self.x1]


class FilteredArea(Area):
    def __init__(self, coords, color_low, color_high, hsv, relative=False, relativeCenterX=False):
        super().__init__(coords, hsv, relative, relativeCenterX)
        self.low = np.array(color_low)
        self.high = np.array(color_high)
	
    def getMask(self):		
        mask = cv2.inRange(self.roi, self.low , self.high);
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        return mask


def removeBlackArea(hsv):
    rows, cols, _ = hsv.shape
    tested_area = (int(rows - rows*0.1), rows, 0, cols)
    black_area = FilteredArea(tested_area, [0,0,0], [10, 255, 5], hsv)
    vector_avg = cv2.reduce(black_area.getMask(), 1, cv2.REDUCE_AVG)
    v_rows, _ = vector_avg.shape
    i = v_rows - 1
    while i > 0 and vector_avg[i, 0] > 5:
        i -= 1
    black_area_height = v_rows - 1 - i	
    return hsv[0:rows-black_area_height, 0:cols]


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
    hsv = removeBlackArea(hsv)

    gym_area  = FilteredArea(config.GYM_AREA,  [0, 0, 210], [180, 80, 255], hsv, relative=True)		
    time_area = FilteredArea(config.TIME_AREA, [0, 0, 230], [180, 60, 255], hsv, relative=True, relativeCenterX=True)
    cp_area   = FilteredArea(config.CP_AREA,   [0, 0, 252], [180, 20, 255], hsv, relative=True)
    poke_area = FilteredArea(config.POKE_AREA, [0, 0, 250], [180, 25, 255], hsv, relative=True)		
    
    areas = [gym_area, time_area, cp_area, poke_area]
    
    gym_pm, time_pm, cp_pm, poke_pm = [Image.fromarray(x.getMask()) for x in areas]

## convert to Pillow format
##    gym_m = cv2.cvtColor(gym_m, cv2.COLOR_GRAY2RGB)
#    gym_pm = Image.fromarray(gym_m)
#
##    time_m = cv2.cvtColor(time_m, cv2.COLOR_BGR2RGB)
#    time_pm = Image.fromarray(time_m)
#
##    cp_m = cv2.cvtColor(cp_m, cv2.COLOR_BGR2RGB)    
#    cp_pm = Image.fromarray(cp_m)
#    
##    poke_m = cv2.cvtColor(poke_m, cv2.COLOR_BGR2RGB)
#    poke_pm = Image.fromarray(poke_m)
#
    gym_text_eng = ""

    with PyTessBaseAPI() as api:
        
        api.SetImage(gym_pm)
        gym_text_eng = api.GetUTF8Text()        
        print("Gym ENG: ", gym_text_eng)
        print(api.AllWordConfidences())

        # for time
        api.SetVariable('tessedit_char_whitelist', 'cp0123456789:')
        api.SetImage(time_pm)
    
        print("Raid Time: ", api.GetUTF8Text())
        print(api.AllWordConfidences())
    
        api.SetVariable('tessedit_char_whitelist', '0123456789')
        api.SetImage(cp_pm)
        print("Boss CP: ", api.GetUTF8Text())
    
        api.SetVariable('tessedit_char_whitelist', 'abcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ')
        api.SetImage(poke_pm)
        print("Poke Name: ", api.GetUTF8Text())
        print(api.AllWordConfidences())


    with PyTessBaseAPI(lang='rus') as api:
        api.SetImage(gym_pm)
        print("Gym RUS: ", api.GetUTF8Text())
        print(api.AllWordConfidences())

main()

