import cv2
import numpy as np
import collections
import math
from scipy import ndimage
from splitter import trim_char


FILE1 = "data/kanji-Gothic/kanji_409.png"
FILE2 = "data/kanji-Mincho/kanji_1.png"
FILE3 = "data/kanji-Mincho/kanji_2.png"

NUM_BINS = 2
NUM_LARGE_BINS = 2
SCALED_SIZE = 48
PDC_AVG = 4

def getSurfBins(img, kps):
  height, width = img.shape
  binW = math.floor(width / NUM_BINS)
  binH = math.floor(height / NUM_BINS)

  bins = np.zeros((NUM_BINS, NUM_BINS))

  for kp in kps:
    x = min(math.floor(kp.pt[0] / binH), NUM_BINS - 1)
    y = min(math.floor(kp.pt[1] / binW), NUM_BINS - 1)
    bins[x, y] += 1

  bins.flatten()
  #print bins
  return bins.flatten()


def orbFeatures(img):
  # scaled = cv2.resize(img, (SCALED_SIZE, SCALED_SIZE))
  orb = cv2.ORB_create(nfeatures = 15)
  kp = orb.detect(img, None)
  print len(kp)
  out = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0))
  cv2.imwrite('keypoints.png', out)
  cv2.waitKey()

  bins = getSurfBins(img, kp)
  return bins

def COG(img):

  cogs = np.zeros(NUM_LARGE_BINS * NUM_LARGE_BINS * 2)
  height, width = img.shape
  for i in range(NUM_LARGE_BINS):
    for n in range(NUM_LARGE_BINS):
      im = img[i * height/NUM_LARGE_BINS:(i+1)*height/NUM_LARGE_BINS - 1, n * width/NUM_LARGE_BINS:(n+1)*width/NUM_LARGE_BINS - 1]
      x_com, y_com = ndimage.measurements.center_of_mass(im)
      x, y = im.shape
      if math.isnan(x_com) or math.isnan(y_com):
        x_com = x / 2
        y_com = y / 2
      cogs[i * NUM_LARGE_BINS * 2 + n * 2] = x_com
      cogs[i * NUM_LARGE_BINS * 2 + n * 2 + 1] = y_com

  return np.array(cogs)
'''
  return np.array(cogs)
  numPoints = 0
  heightTotal = 0
  widthTotal = 0
  for h in range(height):
    for w in range(width):
      if img[h, w] == 0:
        numPoints += 1
        heightTotal += h
        widthTotal += w

  if numPoints != 0:
    widthTotal /= numPoints
    heightTotal /= numPoints
  return np.array((widthTotal, heightTotal))
  '''

def show(img):
  cv2.imshow('title', img)
  cv2.waitKey()

def PDC_diag_features(img, bw = False):
  if bw:
    im_bw = img
  else:
    (thresh, im_bw) = cv2.threshold(
              img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  scaled = cv2.resize(im_bw, (SCALED_SIZE, SCALED_SIZE))

  all_layers = []

  # Whoah https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
  diags = [scaled[::-1,:].diagonal(i)
            for i in range(-scaled.shape[0]+1,scaled.shape[1])]
  diags.extend(scaled.diagonal(i) for i in range(scaled.shape[1]-1,-scaled.shape[0],-1))
  for run in diags:
    start = None
    layers = [0] * 3
    l_i = 0
    for i, pixel in enumerate(run):
      if start == None and pixel == 0:
        start = i
      if start != None and pixel != 0:
        layers[l_i] = i - start
        start = None
        l_i += 1
        if l_i == 3:
          break
    all_layers.append(layers)

  # not sure if this is perfect :/
  l = len(all_layers)
  results = [np.mean(all_layers[row:row+PDC_AVG], axis=0) for row in range(0, l - PDC_AVG / 2, PDC_AVG / 2)]

  return np.concatenate(results).flatten()


def PDC_features(img, bw = False):
  if bw:
    im_bw = img
  else:
    (thresh, im_bw) = cv2.threshold(
              img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # black is 0, white is 255
  scaled = cv2.resize(im_bw, (SCALED_SIZE, SCALED_SIZE))

  CARDINAL_DIRS = [
    # startcol, startrow, drow, dcol, row_order
    (0, 0, 1, 1, 1),
    (0, 0, 1, 1, 0),
    (47, 0, -1, 1, 1),
    (0, 47, 1, -1, 0)
  ]
  directionLayers = []

  for startcol, startrow, drow, dcol, row_order in CARDINAL_DIRS:
    full_row_vals = []
    for i in range(SCALED_SIZE):
      # each row
      l_i = 0
      layers = [0] * 3
      start = None
      for j in range(SCALED_SIZE):
        if row_order == 1:
          pixelVal = scaled[startcol + i * drow, startrow + j * dcol];
        else:
          pixelVal = scaled[startrow + j * dcol, startcol + i * drow];
        if start == None and pixelVal == 0:
          start = j
        if start != None and pixelVal != 0:
          layers[l_i] = abs(abs(dcol * j + drow * (i))-start)
          l_i += 1
          start = None
          if l_i == 3:
            break
      full_row_vals.append(layers)
    directionLayers.append(full_row_vals);

  results = [
      [np.mean(row_vals[i:i+PDC_AVG], axis=0) for i in range(0,SCALED_SIZE - PDC_AVG / 2, PDC_AVG / 2)]
      for row_vals in directionLayers
  ]

  return np.concatenate(results, axis=1).flatten()

def hog(img):
  gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
  gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
  mag, ang = cv2.cartToPolar(gx, gy)

  # quantizing binvalues in (0...16)
  bins = np.int32(NUM_BINS*ang/(2*np.pi))
  # Divide to 4 sub-squares
  bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
  mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
  hists = [np.bincount(b.ravel(), m.ravel(), NUM_BINS) for b, m in zip(bin_cells, mag_cells)]
  hist = np.hstack(hists)
  return hist

def global_features(img):
    height, width = img.shape
    height_width_ratio = 1. * height / width
    x_com, y_com = ndimage.measurements.center_of_mass(img)
    if math.isnan(x_com) or math.isnan(y_com):
        x_com = width / 2
        y_com = height / 2

    row_std = np.mean(np.std(img, axis=0))
    col_std = np.mean(np.std(img, axis=1))

    num_blk_pixels = height * width - (1. * np.sum(img) / 255)
    ratio_filled = num_blk_pixels / (height * width)

    return np.array([
        height_width_ratio,
        x_com,
        y_com,
        row_std,
        col_std,
        ratio_filled,
    ])

def all_features(img, bw = False, classifying = False):
  if not bw:
    (thresh, im_bw) = cv2.threshold(
                img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  else:
    im_bw = img
  # black is 0, white is 255
  im_bw = 255 - trim_char(255 - im_bw)
  if im_bw.size == 0:
    print "empty img, should be impossible"

  scaled = cv2.resize(im_bw, (SCALED_SIZE, SCALED_SIZE))
  feats = np.concatenate((
    #global_features(img),
    PDC_features(scaled, True),
    PDC_diag_features(scaled, True),
    COG(scaled),
    #orbFeatures(scaled_large),
    #hog(scaled_large)
  ))
  return feats


def main():
  img1 = cv2.imread(FILE1, cv2.IMREAD_GRAYSCALE)
  img2 = cv2.imread(FILE1, cv2.IMREAD_GRAYSCALE)
  orbFeatures(img2)
  #img3 = cv2.imread(FILE3, cv2.IMREAD_GRAYSCALE)
  #print PDC_features(img1)
  for a,b in zip(PDC_diag_features(img1), PDC_diag_features(img2)): #, PDC_features(img3)):
    print a
    print b
    #print c
    print "=="

if __name__ == "__main__":
  main()
