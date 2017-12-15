from __future__ import (
    division,
    print_function,
)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch
import skimage.data
from skimage import io

def main():

    # loading image
    img = io.imread('desk.jpg')
    #img = skimage.data.astronaut()
    #io.imshow(img)
    #plt.show()
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.9, min_size=10)
        
    candidates = set()
    for r in regions:
        # excluding same recttangle (with different segments)
        if r['rect'] in candidates:
            continue
            
        # excluding regions smaller than 2000 pixels
        if r['size'] < 3000:
            continue
            
        # distoreed rects
        x, y, w, h = r['rect']
        if w/h > 1.3 or h/w > 1.3:
            continue
        #print("x:{} y:{} w:{} h:{}".format(x, y, w, h))
        candidates.add(r['rect'])
    # draw rectangles on the original iamge
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.imshow(img)
    for x, y, w, h in candidates:
        #print("x:{} y:{} w:{} h:{}".format(x, y, w, h))
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    
    plt.show()
    
    
if __name__=='__main__':
    main()