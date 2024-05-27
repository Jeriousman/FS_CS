import numpy as np
import cv2


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):  
    '''
        랜드마크를 일부러 크게 잡아서 여유있게 만든다
    '''

    lmrks = np.array( lmrks.copy(), dtype=np.int32 )

    # Top of the eye arrays
    bot_l = lmrks[[35, 41, 40, 42, 39]]  ##Where to see which landmark is which?
    bot_r = lmrks[[89, 95, 94, 96, 93]]  ##Where to see which landmark is which?

    # Eyebrow arrays
    top_l = lmrks[[43, 48, 49, 51, 50]]
    top_r = lmrks[[102, 103, 104, 105, 101]]

    # Adjust eyebrow arrays
    lmrks[[43, 48, 49, 51, 50]] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[[102, 103, 104, 105, 101]] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_mask(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Get face mask of image size using given landmarks of person
    """

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)

    points = np.array(landmarks, np.int32)
    convexhull = cv2.convexHull(points)  ##convexHull = https://medium.com/analytics-vidhya/contours-and-convex-hull-in-opencv-python-d7503f6651bc
    cv2.fillConvexPoly(mask, convexhull, 255)  ##https://kali-live.tistory.com/9  points에 저장된 좌표로 이루어진 볼록다각형 or 일반 다각형을 color로 채운다. (fillPoly보다 빠르다.)

    
    return mask


def face_mask_static(image: np.ndarray, landmarks: np.ndarray, landmarks_tgt: np.ndarray, params = None) -> np.ndarray:
    """
    Get the final mask, using landmarks and applying blur
    """
    if params is None:
    
        left = np.sum((landmarks[1][0]-landmarks_tgt[1][0], landmarks[2][0]-landmarks_tgt[2][0], landmarks[13][0]-landmarks_tgt[13][0]))
        right = np.sum((landmarks_tgt[17][0]-landmarks[17][0], landmarks_tgt[18][0]-landmarks[18][0], landmarks_tgt[29][0]-landmarks[29][0]))
        
        offset = max(left, right)
        
        if offset > 6:
            erode = 15
            sigmaX = 15
            sigmaY = 10
        elif offset > 3:
            erode = 10
            sigmaX = 10
            sigmaY = 8
        elif offset < -3:
            erode = -5
            sigmaX = 5
            sigmaY = 10
        else:
            erode = 5
            sigmaX = 5
            sigmaY = 5
    else:
        erode = params[0]
        sigmaX = params[1]
        sigmaY = params[2]
    
    if erode == 15:
        eyebrows_expand_mod=2.7
    elif erode == -5:
        eyebrows_expand_mod=0.5
    else:
        eyebrows_expand_mod=2.0
    landmarks = expand_eyebrows(landmarks, eyebrows_expand_mod=eyebrows_expand_mod)
    
    mask = get_mask(image, landmarks)
    mask = erode_and_blur(mask, erode, sigmaX, sigmaY, True)
    
    if params is None:
        return mask/255, [erode, sigmaX, sigmaY]
        
    return mask/255


def erode_and_blur(mask_input, erode, sigmaX, sigmaY, fade_to_border = True):
    mask = np.copy(mask_input)
    
    if erode > 0:
        kernel = np.ones((erode, erode), 'uint8')
        mask = cv2.erode(mask, kernel, iterations=1)  ##https://www.geeksforgeeks.org/python-opencv-cv2-erode-method/   kernel사이즈만큼 eroding을 진행한다 
    
    else:
        kernel = np.ones((-erode, -erode), 'uint8')
        mask = cv2.dilate(mask, kernel, iterations=1) ##https://nicewoong.github.io/development/2018/01/05/erosion-and-dilation/ erode의 반대개념
        
    if fade_to_border:
        clip_size = sigmaY * 2
        mask[:clip_size,:] = 0  ##이미지를 가장자리부분을  모두 0을 넣어서 마스크 array를 만들어서 fading하게 만들자는 것 같다  
        mask[-clip_size:,:] = 0
        mask[:,:clip_size] = 0
        mask[:,-clip_size:] = 0
    
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX = sigmaX, sigmaY = sigmaY)  ##https://deep-learning-study.tistory.com/144 마스크를 일부러 blur하게 만들어서 모호하게한다
        
    return mask



