import cv2
import numpy as np
import skimage.filters as filters
from matplotlib import pyplot as plt
from PIL import Image
import concurrent.futures
import time
from multiprocessing import Process, Manager
from skimage import feature
from scipy.ndimage import distance_transform_edt
import random



# if __name__ == "__main__":
#     img = cv2.imread('plane.jpg',0)
#     cv2.imshow('image', img)
#     cv2.waitKey(0)



def get8n(x, y, shape,processed):
    out = []
    maxx = shape[0]-1
    maxy = shape[1]-1

    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    if not (outx,outy) in processed:
        out.append((outx,outy))

    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))

    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    if not (outx, outy) in processed:
        out.append((outx, outy))
    print(out)
    return out

def region_growing(img, seed, outimg,processed):
    lista = []
    c1 = float(np.min(img))
    c2 = float(np.max(img))
    contrast = (c2 - c1) / (c2 + c1)
    print('kontrast',c1,c2,contrast)
    contrast2 = img.std()
    print('kontrast2: ',contrast2)
    lista.append((seed[0],seed[1]))
    reg=int(img[seed[0], seed[1]])
    size=1
    color = list(np.random.choice(range(256), size=3))
    while(len(lista) > 0):
        pix = lista[0]
        outimg[pix[0], pix[1]] = color
        q=img[pix[0],pix[1]]
        for coord in get8n(pix[0], pix[1], img.shape,processed):
            # if ((-50*contrast)<(img[coord[0], coord[1]] - int(img[seed[0],seed[1]]))<(50*contrast)): # kontrast reczny z zakresu wartosci pikseli
            # if ((-0.4 * contrast2) < (img[coord[0], coord[1]] - int(img[seed[0], seed[1]])) < (0.4 * contrast2)): #kontrast z funkcji
            w=int(img[coord[0], coord[1]])
            e=int(img[seed[0], seed[1]])
            r=w-e
            if (-30 < (int(img[coord[0], coord[1]]) - reg) < 30):  # srednia z regionu
                outimg[coord[0], coord[1]] = color
                reg=(reg*size+int(img[coord[0], coord[1]]))/(size+1)
                size = size + 1
                if not coord in processed:
                    lista.append(coord)
                processed.append(coord)
        lista.pop(0)
        cv2.imshow("progress",outimg)
        cv2.waitKey(1)
    return outimg


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Seed: ' + str(x) + ', ' + str(y), image[y,x])
        clicks.append((y,x))
    if event == cv2.EVENT_RBUTTONDOWN:
        print(' Removed Last Seed')
        clicks.pop()

if __name__ == '__main__':

    clicks = []
    zx=[]
    #manager = Manager()
    #processed = manager.list()
    processed = []

    image = cv2.imread('bird.jpg', 0)

    x = image.shape[0]
    y = image.shape[1]
    if x>250 or y>250:
        scale_percent=(250/image.shape[0])*100
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (width,height), interpolation = cv2.INTER_AREA)
        image=np.array(image)

    edges = cv2.Canny(image, 130, 200)
    indices = np.where(edges != [0])
    coordinates = list(zip(indices[0], indices[1]))

    # for z in coordinates:
    #     print(z)
    # edges = feature.canny(image, sigma=3)

    print('długosc',len(coordinates))
    # cv2.imshow('edges',edges)
    rng=random.choice(coordinates)
    print('choice is: ',rng)

    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #ret, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('edges',thresh)

    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            clicks.append((cY, cX))
        else:
            cX, cY = 0, 0
        #while(len(clicks)<6):
            #zx.append((cX,cY),image[cY,cX])

            #clicks.append((cX, cY))
        v=image[cY,cX]
        # cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.putText(image, "centroid", (cX - 25, cY - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (127, 128, 200), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    # dt=distance_transform_edt(~edges)
    # plt.imshow(dt)
    # plt.show()
    # peakid = feature.peak_local_max(dt, indices=True, min_distance=1)
    # print(peakid)
    # localmax=feature.peak_local_max(dt,indices=False,min_distance=1)
    # plt.imshow(dt,cmap='gray')
    # plt.show()
    # plt.plot(peakid[:,1],peakid[:,0],'r.')
    # plt.imshow(dt)
    # plt.show()

    #counts,bins,bars=plt.hist(image.ravel(),255,[0,256])
    # ret, img = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)




    #print(np.argmax(counts))
    #print(bins[np.argmax(counts)])
    #np.where(image == bins[np.argmax(counts)], image, image)
    # clicks=np.extract(image==bins[np.argmax(counts)], image)
    # print(clicks)

    #pixels = np.where(image ==(bins[np.argmax(counts)]))
    #print(pixels)

    outimg = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)


    '''na pzrycisk'''
    cv2.namedWindow('Input')
    #cv2.setMouseCallback('Input', on_mouse, 0, )
    cv2.imshow('Input', image)
    cv2.waitKey(0)

    rest=np.where(np.all(outimg == [0,0,0], axis=-1))
    listOfCoordinates = list(zip(rest[0], rest[1]))
    #print('gdzie ty kurwa jestes',listOfCoordinates[0])
    #outimg = np.zeros_like(image)

    # i=0
    start_time = time.time()
    #while(listOfCoordinates!=[]):
    # for i in clicks:
    #
    #     #seed = listOfCoordinates[0]
    #     seed=i
    #     out = region_growing(image, seed,outimg,processed)


    # clicks=[]
    # clicks.append(rng)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results=[executor.submit(region_growing,image,seed,outimg,processed) for seed in clicks]

        for f in concurrent.futures.as_completed(results):
            outimg=outimg+f.result()

    end_time = time.time()
    time_taken = end_time - start_time
    print('time taken to complete: ',time_taken)
    #outimg = cv2.resize(outimg, (y, x), interpolation=cv2.INTER_AREA)
    cv2.imshow('Region Growing', outimg)
    cv2.waitKey()
    edges = cv2.Canny(outimg, 100, 200)
    cv2.imshow('edges',edges)
    cv2.waitKey()
    cv2.destroyAllWindows()