"""ex1 image processing - daniel.hazan
part1:reading, displaying and converting images from RGB represantation to YIQ and backforward.
part2: calculating equalized histogram of images
part 3: calculating quantization algorithm for YIQ and RGB images
"""

import numpy as np
from skimage.color import rgb2grey
from skimage.color import rgb2yiq as bla
from scipy.misc import imread
import matplotlib.pyplot as plt

def read_image(filename, representation):
    image = imread(filename)
    if(len(image.shape)<3):
        #the third dimension which indicates the colour-channels
        #is missing, meaning its a gray-scale image

        #convert it to float64
        im_float = image.astype(np.float64)
        im_float /= 255
        return im_float
    if(len(image.shape) == 3):
        #RGB image

        if(representation ==1):
            #convert to gray-scale
            im_g = rgb2grey(image)
            im_g = im_g.astype(np.float64)


            return im_g
        if representation ==2:
            im_f =  image.astype(np.float64)
            im_f /= 255

            return im_f

def imdisplay(filename,representation):

    image = read_image(filename,representation)

    if(representation ==2):
        plt.imshow(image)

    if(representation == 1):
        plt.imshow(image, cmap= 'gray')

    plt.show()

def rgb2yiq(imRGB):

    redChannel = imRGB[:,:,0]
    greenChannel = imRGB[:,:,1]
    blueChannel = imRGB[:,:,2]

    y = 0.299*redChannel + 0.587*greenChannel + blueChannel*0.144
    i = 0.596*redChannel  - 0.275*greenChannel - blueChannel*0.321
    q  = 0.212*redChannel - 0.523*greenChannel + blueChannel*0.311

    imYIQ = imRGB[:,:,:].copy()



    imYIQ[:,:,0] = y
    imYIQ[:,:,1] = i
    imYIQ[:,:,2] = q

    return imYIQ


def yiq2rgb(imYIQ):

    y= imYIQ[:,:,0]
    i= imYIQ[:,:,1]
    q= imYIQ[:,:,2]

    r = y+0.956*i +0.621*q
    g=y-0.272*i-0.647*q
    b=y-1.106*i+1.703*q
    imRGB = imYIQ[:,:,:].copy()
    imRGB[:,:,0] = r
    imRGB[:,:,1] = g
    imRGB[:,:,2] = b

    return imRGB

def histogram_equalize(im_orig):
    if(len(im_orig.shape)<3):
        equaIm, origHisto,equaHisto = funcHisto(im_orig)
    if(len(im_orig.shape) ==3):
        yiqIm = rgb2yiq(im_orig)
        yChannel = yiqIm[:,:,0].copy()
        newY,origHisto,equaHisto= funcHisto(yChannel)


        yiqIm[:,:,0] = newY/255 # remember to normalize the new equalized values of Y channel!
        equaIm = yiq2rgb(yiqIm)

    return [equaIm,origHisto,equaHisto]

def funcHisto(im_orig):

    imageOrig_histo, bins = np.histogram(im_orig.flatten(),256,normed=True)
    comulativeDistF = imageOrig_histo.cumsum()
    comulativeDistF = 255*comulativeDistF/comulativeDistF[-1]#normalize the comulative histo
    #verify that the minimal value is 0 and the maximal value is 255
    nonZeroF = np.nonzero(comulativeDistF)[0][0]
    comulativeDistF = (comulativeDistF-comulativeDistF[nonZeroF])/(comulativeDistF[255]-comulativeDistF[nonZeroF])*255
    comulativeDistF = np.clip(comulativeDistF, 0, 255)

    #round the values to get an integer
    comulativeDistF = np.around(comulativeDistF)
    #convert the comulative histogram matrix to equalized image using linear interpolation
    equalIm = np.interp(im_orig.flatten(), bins[:-1],comulativeDistF)
    equalIm = equalIm.reshape(im_orig.shape)


    equaHisto = np.histogram(equalIm.flatten(),256,normed=True)

    return equalIm,imageOrig_histo,equaHisto[0]


def quantize(im_orig,n_quant,n_iter):

    if(len(im_orig.shape) ==3):
        yiqIm = rgb2yiq(im_orig)
        imInt = yiqIm[:,:,0].copy()
    else:
        imInt = im_orig.copy()

    hist ,bins = np.histogram(imInt,256)

    #check if the image already quantized
    if(len(np.unique(imInt)) == n_quant):
        im_quant = im_orig
        error = 0

    #search for initial division of Z

    zArray = initializeZ(hist,n_quant)

    #now iterate Z,Q and the error
    q = [0 for i in range(len(zArray)-1)]
    errorArray = []
    ifConverge = False

    for i in range(n_iter):
        q = getqVal(zArray,hist,q)
        q = np.round(q).astype(int)

        #error cal.
        error = 0

        for i in range(len(q) ):

            for k in range(zArray[i],zArray[i+1] + 1):
                error += (q[i] -k)**2 * hist[k]
        errorArray.append(error)
        if(ifConverge):
            break

        zArray,ifConverge = getzVal(q,zArray)

    #after convergence, convert into quantized image using linear interpolation-->
    arrayQuantize = []

    for k in range(len(q)):
        tempArray = []
        for i in range(zArray[k],zArray[k+1]):
            tempArray.append(q[k])
        arrayQuantize.extend(tempArray)

    if(len(im_orig.shape) ==3):
        #use only y channel
        imageQ = np.interp(imInt,np.linspace(0,1,255),arrayQuantize)
        yiqIm[:,:,0] = imageQ/255

        return yiq2rgb(yiqIm),errorArray

    imageQ = np.interp(imInt,np.linspace(0,1,255),arrayQuantize)



    return imageQ,errorArray



def initializeZ(hist,n_quant):
    """initialize z values """

    comulativeH = hist.cumsum()

    dividePixelsToQuants = comulativeH[-1]/(n_quant)
    equalBins = np.linspace(dividePixelsToQuants,comulativeH[-1],n_quant)

    #take the indexes of the first values of commulative histogram which succeeds equalBins
    diffs = [0] + [np.amax(np.where(comulativeH <= equalBins[k])) for k in range(n_quant)]

    if(diffs[-1] != 255):
        diffs[-1] = 255

    return diffs

def getqVal(zAray,hist,q):
    """calculate q according to z using formula """


    for k in range(len(q) ):
        mone=0
        mechane = 0

        for i in range(zAray[k],zAray[k+1] +1):

            mone += (i*hist[i] )
            mechane += hist[i]
        if(zAray[k] == zAray[k+1]):
            q[k] = zAray[k]

        elif(mone==0 or mechane ==0):
            q[k] = 0

        else:

            q[k] = mone/mechane


    return q

def getzVal(Q,zArray):
    """calculate z segments from Q values using formula, and checking if
    there is convergence. returns the new array and and boolean expression indicating the convergence"""
    count = 0
    for i in range(1,len(zArray)-1):
        zCheck = zArray[i]
        zArray[i] = np.ceil((Q[i-1] +Q[i])/2).astype(int)

        if(zCheck == zArray[i]):
            count += 1


    if(count == len(zArray) -2): #excluding first and last value of zArray

        return zArray,True


    return zArray ,False
