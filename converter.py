import numpy as np
import math, cv2

def toBinary(a):
  l,m=[],[]
  for i in a:
    l.append(ord(i))
  for i in l:
    binary = bin(i)
    if len(binary) < 9:
      m += [0]*(9-len(binary))
    for k in binary:
      if k == 'b':
        continue
      m.append(int(k))
  return m

def toString(a):
  l=[]
  m=""
  for i in a:
    b=0
    c=0
    k=int(math.log10(i))+1
    for j in range(k):
      b=((i%10)*(2**j))   
      i=i//10
      c=c+b
    l.append(c)
  for x in l:
    m=m+chr(x)
  return m

def rgb_to_hue(b, g, r):
    if (b == g == r):
        return 0

    angle = 0.5 * ((r - g) + (r - b)) / math.sqrt(((r - g) ** 2) + (r - b) * (g - b))
    if b <= g:
        return math.acos(angle) * 180 / math.pi
    else:
        return (2 * math.pi - math.acos(angle)) * 180 / math.pi


def rgb_to_intensity(b, g, r):
    val = (b + g + r) / 3.

    return val * 255


def rgb_to_saturity(b, g, r):
    if r + g + b != 0:
        return (1. - 3. * np.min([r, g, b]) / (r + g + b)) * 100
    else:
        return 0

def rgbToHSI(img, img_shape):
    height, width = img_shape[0], img_shape[1]
    I = np.zeros((height, width))
    S = np.zeros((height, width))
    H = np.zeros((height, width))

    for i in range(height):
      for j in range(width):
        b = img[i][j][0] / 255.
        g = img[i][j][1] / 255.
        r = img[i][j][2] / 255.
        H[i][j] = rgb_to_hue(b, g, r)
        S[i][j] = rgb_to_saturity(b, g, r)
        I[i][j] = rgb_to_intensity(b, g, r)

    return cv2.merge((H,S,I))

def hsiToRGB(hsi, img_shape):
    H = hsi[:, :, 0] * math.pi / 180
    S = hsi[:, :, 1] / 100
    I = hsi[:, :, 2] / 255
    height, width = img_shape[0], img_shape[1]
    new_image = np.zeros((height, width, 3), dtype=np.uint8)

    def convert(h, s, i): 
      b, g, r = 0, 0, 0
      h = math.degrees(h)
      if 0 <= h <= 120 :
          b = i * (1 - s)
          r = i * (1 + (s * math.cos(math.radians(h)) / math.cos(math.radians(60) - math.radians(h))))
          g = i * 3 - (r + b)
      elif 120 < h <= 240:
          h -= 120
          r = i * (1 - s)
          g = i * (1 + (s * math.cos(math.radians(h)) / math.cos(math.radians(60) - math.radians(h))))
          b = 3 * i - (r + g)
      elif 0 < h <= 360:
          h -= 240
          g = i * (1 - s)
          b = i * (1 + (s * math.cos(math.radians(h)) / math.cos(math.radians(60) - math.radians(h))))
          r = i * 3 - (g + b)
      return [b, g, r]

    for i in range(height):
      for j in range(width):
        bgr_tuple = convert(H[i][j], S[i][j], I[i][j])

        new_image[i][j][0] = round(bgr_tuple[0] * 255.)
        new_image[i][j][1] = round(bgr_tuple[1] * 255.)
        new_image[i][j][2] = round(bgr_tuple[2] * 255.)
    
    return new_image

# generate msg for embedding
def genMsg(bi_secret_msg, img_shape):
  dummy_arr = np.zeros((img_shape[0]*img_shape[1] - len(bi_secret_msg)))
  new_secret_msg = np.append(np.array([bi_secret_msg]), dummy_arr).reshape(img_shape)

  return new_secret_msg

# embed msg in hsi img (intensity channel)
def embed(cover_img, secret_msg, k=1):
    stego = np.array(cover_img, copy=True)
    mask = 256 - 2**k
    stego[:,:,2] = stego[:,:,2].astype(int) & mask | secret_msg.astype(int)
    return stego

# extract embedded msg from hsi
def extract(stego, k=1):
    iPlane = stego[:,:,2].astype(int)
    output = []
    mask = 2**k - 1
    stego_shape = iPlane.shape
    secret_msg = (iPlane & mask)
    secret_msg = secret_msg.flatten('K')
    for byte in np.split(secret_msg, stego_shape[0]*stego_shape[1]/8):
      byte_value = int(''.join(byte.astype(str)))
      if byte_value == 0:
        break
      output += [byte_value]
    # cv2.imwrite('extracted.png', secret_msg)

    return output