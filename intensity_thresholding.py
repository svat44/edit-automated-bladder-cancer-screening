# INTENSITY THRESHOLDING CODE WRITTEN & CREDIT TO SIDDHARTHA VATSA | GITHUB svat44
# For "EDIT AI Bladder Cancer Detection through Advanced Deep Learning Models" | 2026 | File Initial Commit 7-3-2026

"""
AI Use Disclaimer: This code was debugged and cleaned with the assistance of AI. 
The author takes full responsibility for the content and functionality of this code.
AI was not used in the main construction, framework, or development of any image segementation and evaluation techniques. 
"""
abstract = None

import time
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def thresholding(currimgnum):  # gather the selected cell image
    root = os.getcwd()
    imgPath = np.load(os.path.join(root, "imagedata/X/" + str(currimgnum) + ".npy")) # load the image data from the specified path
    img = np.transpose(imgPath, (1, 2, 0)) # transpose the image data to the correct shape for processing
    imgGray = cv.cvtColor((img*255).astype(np.uint8), cv.COLOR_BGR2GRAY) # convert to grayscale
    imgNorm = imgGray.astype(np.float32) / 255.0 # normalize the values to [0, 1]: allows for fine-tuning and works better for N-C

    hist = cv.calcHist([imgGray], [0], None, [255], [0, 255]) # histogram is scaled from 0-255 to neglect the background
    plt.figure()
    plt.plot(hist)
    plt.xlabel("bins")
    plt.ylabel("#px") # Number of pixels in each bin (intensity value)

    threshOpt = [cv.THRESH_BINARY] #, cv.THRESH_BINARY_INV, cv.THRESH_TOZERO, cv.THRESH_TOZERO_INV, cv.THRESH_TRUNC
    thresNames = ["BINARY", "BINARY_INV", "TOZERO", "TOZERO_INV", "TRUNC"]

    plt.figure()
    plt.subplot(231)
    plt.imshow(imgNorm, cmap="gray")

    THRESHOLD = 0.4
    TOLERANCE = 0.005
    thr_step = 0.01
    last_direction = None

    def gatherThreshold(rt, gtVal):
        nonlocal THRESHOLD, thr_step, last_direction
        if (rt - gtVal) > TOLERANCE:
            direction = -1
        elif (gtVal - rt) > TOLERANCE:
            direction = 1
        else:
            return  # close enough, don't move

        if last_direction is not None and direction != last_direction:
            thr_step /= 2  # we overshot -> take smaller steps next time
        
        THRESHOLD += direction * thr_step
        last_direction = direction

    def doThresholding():
        last_idx = None
        for idx in range(len(threshOpt)): # process of thresholding and outputting the results
            plt.subplot(2, 3, idx + 2)
            _, thres = cv.threshold(imgNorm, THRESHOLD, 1, threshOpt[idx])
            plt.imshow(thres, cmap="gray", vmin=0, vmax=1)
            plt.title(thresNames[idx])
            last_idx = idx
        return last_idx
                    

    def calcRatio_p(n_pred,c_pred):
        return n_pred / (n_pred + c_pred) if (n_pred + c_pred) > 0 else float('inf')
    def calcRatio_g(n_gt,c_gt):
        return n_gt / (n_gt + c_gt) if (n_gt + c_gt) > 0 else float('inf')

    cyto = np.count_nonzero((imgNorm >= THRESHOLD) & (imgNorm < 0.8)) # cells with an intensity greater than 0.4 and less than 0.8 are considered cytoplasm
    nuc = np.count_nonzero(imgNorm < THRESHOLD) # cells with an intensity less than 0.4 are considered nucleus
    
    ratio = calcRatio_p(nuc, cyto)

    # use the first thresholding method name for initial print
    print(f"REG: {thresNames[0]}: cytoplasm={cyto}, nucleus={nuc}, N/(N+C)={ratio:.3f}") #output of the result of the computer-generated mask
    
    mask = np.load(os.path.join(root, "imagedata/y/" + str(currimgnum) + ".npy")) # load the ground truth mask for comparison




    nuc_gt = np.count_nonzero(mask == 2)
    cyto_gt = np.count_nonzero(mask == 1)

    ratio_gt = calcRatio_g(nuc_gt, cyto_gt)

    print(f"GT: nucleus={nuc_gt}, cytoplasm={cyto_gt}, N/(N+C)={ratio_gt:.3f}") #output of the result of the ground truth mask
    
    """Forever Loop Safeguard"""
    guard = 100
    count = 0

    while abs(ratio - ratio_gt) > TOLERANCE and count < guard: # adjust the threshold until the ratio of nucleus to cytoplasm is within 5% of the ground truth
        gatherThreshold(ratio, ratio_gt)
        last_idx = doThresholding()
        cyto = np.count_nonzero((imgNorm >= THRESHOLD) & (imgNorm < 0.8))
        nuc = np.count_nonzero(imgNorm < THRESHOLD)
        ratio = calcRatio_p(nuc, cyto)
        name = thresNames[last_idx] if last_idx is not None else thresNames[0]
        print(f"REG: {name}: cytoplasm={cyto}, nucleus={nuc}, N/(N+C)={ratio:.3f}")

        count += 1
    
    print(f"GT: nucleus={nuc_gt}, cytoplasm={cyto_gt}, N/(N+C)={ratio_gt:.3f}") #output of the result of the ground truth mask

    #plt.show()

    predicted_mask = np.zeros_like(mask)
    predicted_mask[(imgNorm >= THRESHOLD) & (imgNorm < 0.8)] = 1  # cytoplasm
    predicted_mask[imgNorm < THRESHOLD] = 2  # nucleus
    
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_mask == 1, cmap="gray", vmin=0, vmax=1)
    plt.title("Cytoplasm")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask == 2, cmap="gray", vmin=0, vmax=1)
    plt.title("Nucleus")
    #plt.show()
    

    def iou(pred, gt, class_value):
        predicted_c = (pred == class_value)
        gt_c = (gt == class_value)
        intersection = np.logical_and(predicted_c, gt_c).sum()
        union = np.logical_or(predicted_c, gt_c).sum()
        return intersection / union if union > 0 else float('nan')
    
    nucleusIOU = iou(predicted_mask, mask, 2)
    cytoplasmIOU = iou(predicted_mask, mask, 1)

    overlay = np.zeros((*mask.shape, 3))
    overlay[(predicted_mask == 2) & (mask == 2)] = [1, 1, 0]   # yellow = correct nucleus overlap
    overlay[(predicted_mask == 2) & (mask != 2)] = [1, 0, 0]   # red = predicted nucleus, but wrong
    overlay[(predicted_mask != 2) & (mask == 2)] = [0, 1, 0]   # green = missed real nucleus

    plt.imshow(overlay)
    plt.title("Nucleus overlap: yellow=correct, red=false positive, green=missed")
    #plt.show()

    # return IOUs as a single tuple with values rounded to 3 decimal places
    return (round(nucleusIOU, 3), round(cytoplasmIOU, 3))

if __name__ == "__main__":
    start = time.time()
    iouNucSum = []
    iouCytoSum = []
    for i in range(200): 
        print(f"Processing image {i}")
        ious = thresholding(i)
        plt.close('all')  # Close all figures to prevent memory issues
        iouNucSum.append(ious[0])
        iouCytoSum.append(ious[1])

    end = time.time()
    print(f"Total time taken: {round(end - start, 2)} seconds")
    print(f"Mean IOU of Nucleus: {round(np.nanmean(iouNucSum), 3)}")
    print(f"Mean IOU of Cytoplasm: {round(np.nanmean(iouCytoSum), 3)}")