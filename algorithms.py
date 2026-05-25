import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import pywt
import math
from scipy.stats import entropy

# Function to preprocess the image
def preprocess_image(image, conversion='none', resize_width=256, resize_height=256):
    """
    Applies preprocessing to the image based on user selections.
    - conversion: 'none', 'grayscale', or 'ycbcr'
    - resize_width, resize_height: target size for resizing
    Returns the preprocessed image.
    """
    if image is None:
        return None

    # Apply conversion
    if conversion == 'grayscale':
        processed = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif conversion == 'ycbcr':
        processed = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        processed = image.copy()

    # Resize
    processed = cv2.resize(processed, (resize_width, resize_height))

    return processed

# Function to apply GLCM algorithm
def apply_glcm(image):
    """
    Applies Gray Level Co-occurrence Matrix (GLCM) to the grayscale image.
    Extracts features: Contrast, Energy, Homogeneity, Correlation, Entropy.
    Returns a dictionary of features and the GLCM matrix.
    """
    if image is None:
        return None, "Image must be grayscale for GLCM."

    # Convert to grayscale if color (or extract Y channel for YCbCr)
    if len(image.shape) == 3:
        # If it's a 3-channel image, extract the first channel (Y for YCbCr, or convert from RGB)
        image = image[:, :, 0]  # Extract Y channel
    
    # Ensure image is in 8-bit format (0-255)
    image = np.uint8(image)

    # Define GLCM parameters
    distances = [1]  # Pixel distance
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # 0째, 45째, 90째, 135째

    # Compute GLCM
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    # Extract GLCM features
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    # Compute entropy manually
    glcm_prob = glcm[:, :, 0, 0]  # Consider first direction
    glcm_prob = glcm_prob[glcm_prob > 0]  # Remove zero probabilities to avoid log(0)
    glcm_entropy = entropy(glcm_prob.flatten())

    features = {
        'Contrast': contrast,
        'Energy': energy,
        'Homogeneity': homogeneity,
        'Correlation': correlation,
        'Entropy': glcm_entropy
    }

    return features, glcm

# Function to apply DCT algorithm
def apply_dct(image):
    """
    Applies Discrete Cosine Transform (DCT) to the image.
    Returns the DCT coefficients.
    """
    if image is None:
        return None

    # Extract Y channel if color image (for YCbCr) or convert to grayscale
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Extract Y channel for YCbCr

    # Convert to float32
    image_float = np.float32(image)

    # Apply DCT
    dct = cv2.dct(image_float)

    return dct

# Function to apply DWT algorithm
def apply_dwt(image):
    """
    Applies Discrete Wavelet Transform (DWT) to the image using Haar wavelet.
    Decomposes into LL, LH, HL, HH subbands.
    Returns the coefficients.
    """
    if image is None:
        return None

    # Extract Y channel if color image (for YCbCr)
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Extract Y channel for YCbCr

    # Apply DWT
    coeffs = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs

    return LL, LH, HL, HH

# Function to apply LBP algorithm
def apply_lbp(image):
    """
    Applies Local Binary Pattern (LBP) to the image.
    Returns the LBP image and its histogram.
    """
    if image is None:
        return None, None

    # Extract Y channel if color image (for YCbCr)
    if len(image.shape) == 3:
        image = image[:, :, 0]  # Extract Y channel for YCbCr

    # Apply LBP
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')

    # Compute histogram
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59), range=(0, 58))

    return lbp, hist

# Function to apply Sobel edge detection
def apply_sobel(image):
    """
    Applies Sobel edge detection to the image.
    Returns the Sobel magnitude image and gradient features.
    """
    if image is None:
        return None, None

    # Extract Y channel if color image (for YCbCr)
    if len(image.shape) == 3:
        gray = image[:, :, 0]  # Extract Y channel for YCbCr
    else:
        gray = image

    # Compute Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute magnitude
    sobel = cv2.magnitude(sobelx, sobely)

    # Normalize for display
    sobel_display = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Extract features
    features = {
        'Mean Gradient': float(np.mean(sobel)),
        'Std Gradient': float(np.std(sobel)),
        'Max Gradient': float(np.max(sobel)),
        'Min Gradient': float(np.min(sobel))
    }

    return sobel_display, features

# Function to apply SIFT keypoint detection
def apply_sift(image):
    """
    Applies Scale-Invariant Feature Transform (SIFT) to the image.
    Returns keypoint visualization and basic keypoint features.
    """
    if image is None:
        return None, None

    # Convert to grayscale if color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    if not hasattr(cv2, "SIFT_create"):
        return None, "SIFT is not available in this OpenCV installation."

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    keypoint_image = cv2.drawKeypoints(
        image if len(image.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    if not keypoints:
        features = {
            'Keypoints Count': 0,
            'Descriptor Size': 0,
            'Mean Keypoint Size': 0.0,
            'Mean Response': 0.0
        }
        return keypoint_image, features

    sizes = [kp.size for kp in keypoints]
    responses = [kp.response for kp in keypoints]
    descriptor_size = 0 if descriptors is None else descriptors.shape[1]

    features = {
        'Keypoints Count': len(keypoints),
        'Descriptor Size': descriptor_size,
        'Mean Keypoint Size': float(np.mean(sizes)),
        'Mean Response': float(np.mean(responses))
    }

    return keypoint_image, features


def apply_sift_copy_move(
    image,
    ratio_thresh=0.75,
    max_display_matches=200,
    min_sep_frac=0.04,
    knn_k=5,
):
    """
    Copy-move style visualization: match SIFT descriptors within the same image.
    Keeps pairs that pass Lowe's ratio test and lie far enough apart in pixel space
    (to ignore trivial local neighbors). Draws yellow lines between matches with
    red circles at one end and green at the other (BGR drawing, returned as RGB).
    """
    if image is None:
        return None, "No image for copy-move detection."

    if not hasattr(cv2, "SIFT_create"):
        return None, "SIFT is not available in this OpenCV installation."

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        gray = image
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    min_pixel_dist = max(16.0, min_sep_frac * float(min(h, w)))

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 3:
        return None, "Not enough SIFT features for copy-move matching."

    k = min(knn_k, len(descriptors))
    if k < 3:
        return None, "Not enough descriptors for k-NN matching."

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn = bf.knnMatch(descriptors, descriptors, k=k)

    candidates = []
    for matches in knn:
        if len(matches) < 3:
            continue
        qi = matches[0].queryIdx
        non_self = [m for m in matches if m.trainIdx != qi]
        if len(non_self) < 2:
            continue
        m1, m2 = non_self[0], non_self[1]
        if m1.distance >= ratio_thresh * m2.distance:
            continue
        i, j = int(m1.queryIdx), int(m1.trainIdx)
        if i == j:
            continue
        pt_i = keypoints[i].pt
        pt_j = keypoints[j].pt
        sep = math.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1])
        if sep < min_pixel_dist:
            continue
        candidates.append((min(i, j), max(i, j), float(m1.distance)))

    best_by_pair = {}
    for a, b, dist in candidates:
        prev = best_by_pair.get((a, b))
        if prev is None or dist < prev:
            best_by_pair[(a, b)] = dist

    ordered = sorted(best_by_pair.items(), key=lambda x: x[1])[:max_display_matches]

    if not ordered:
        return None, (
            "No separated duplicate-feature pairs found "
            "(try copy-moved content or adjust preprocessing size)."
        )

    vis = bgr.copy()
    yellow = (0, 255, 255)
    red = (0, 0, 255)
    green = (0, 255, 0)
    for (i, j), _dist in ordered:
        pt1 = (int(round(keypoints[i].pt[0])), int(round(keypoints[i].pt[1])))
        pt2 = (int(round(keypoints[j].pt[0])), int(round(keypoints[j].pt[1])))
        cv2.line(vis, pt1, pt2, yellow, 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt1, 3, red, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt2, 3, green, thickness=-1, lineType=cv2.LINE_AA)

    stats = {
        "Copy-move pairs drawn": float(len(ordered)),
        "Keypoints": float(len(keypoints)),
        "Min point separation (px)": float(min_pixel_dist),
    }
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), stats


def apply_sift_block_structure(
    image,
    grid_rows=4,
    grid_cols=4,
    ratio_thresh=0.75,
    max_display_matches=200,
    min_sep_frac=0.04,
    knn_k=5,
):
    """
    SIFT on a regular grid of image blocks (default 4x4), then same-image descriptor
    matching (copy-move style). Visualization matches a typical block forensics view:
    blue grid, red keypoint markers, green lines between matched pairs.
    """
    if image is None:
        return None, "No image for block SIFT."

    if not hasattr(cv2, "SIFT_create"):
        return None, "SIFT is not available in this OpenCV installation."

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        gray = image
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]
    min_pixel_dist = max(16.0, min_sep_frac * float(min(h, w)))

    if grid_rows < 1 or grid_cols < 1:
        return None, "Grid dimensions must be at least 1."

    cell_h = h // grid_rows
    cell_w = w // grid_cols
    if cell_h < 8 or cell_w < 8:
        return None, "Image too small for the requested block grid."

    sift = cv2.SIFT_create()
    all_keypoints = []
    desc_chunks = []

    for br in range(grid_rows):
        for bc in range(grid_cols):
            y0 = br * cell_h
            x0 = bc * cell_w
            y1 = h if br == grid_rows - 1 else (br + 1) * cell_h
            x1 = w if bc == grid_cols - 1 else (bc + 1) * cell_w
            roi = gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            kp_block, desc_block = sift.detectAndCompute(roi, None)
            if not kp_block or desc_block is None:
                continue
            for kp in kp_block:
                all_keypoints.append(
                    cv2.KeyPoint(
                        float(kp.pt[0] + x0),
                        float(kp.pt[1] + y0),
                        kp.size,
                        kp.angle,
                        kp.response,
                        kp.octave,
                        kp.class_id,
                    )
                )
            desc_chunks.append(desc_block)

    if not all_keypoints or not desc_chunks:
        return None, "No SIFT features found inside any block."

    descriptors = np.vstack(desc_chunks)
    if len(all_keypoints) != len(descriptors):
        return None, "Internal error: keypoint count does not match descriptors."

    k = min(knn_k, len(descriptors))
    ordered = []
    if k >= 3:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        knn = bf.knnMatch(descriptors, descriptors, k=k)

        candidates = []
        for matches in knn:
            if len(matches) < 3:
                continue
            qi = matches[0].queryIdx
            non_self = [m for m in matches if m.trainIdx != qi]
            if len(non_self) < 2:
                continue
            m1, m2 = non_self[0], non_self[1]
            if m1.distance >= ratio_thresh * m2.distance:
                continue
            i, j = int(m1.queryIdx), int(m1.trainIdx)
            if i == j:
                continue
            pt_i = all_keypoints[i].pt
            pt_j = all_keypoints[j].pt
            sep = math.hypot(pt_i[0] - pt_j[0], pt_i[1] - pt_j[1])
            if sep < min_pixel_dist:
                continue
            candidates.append((min(i, j), max(i, j), float(m1.distance)))

        best_by_pair = {}
        for a, b, dist in candidates:
            prev = best_by_pair.get((a, b))
            if prev is None or dist < prev:
                best_by_pair[(a, b)] = dist

        ordered = sorted(best_by_pair.items(), key=lambda x: x[1])[:max_display_matches]

    vis = bgr.copy()
    blue = (255, 0, 0)
    red = (0, 0, 255)
    green = (0, 255, 0)

    for r in range(1, grid_rows):
        y = r * cell_h
        cv2.line(vis, (0, y), (w - 1, y), blue, 1, lineType=cv2.LINE_AA)
    for c in range(1, grid_cols):
        x = c * cell_w
        cv2.line(vis, (x, 0), (x, h - 1), blue, 1, lineType=cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), blue, 1, lineType=cv2.LINE_AA)

    for (i, j), _dist in ordered:
        pt1 = (int(round(all_keypoints[i].pt[0])), int(round(all_keypoints[i].pt[1])))
        pt2 = (int(round(all_keypoints[j].pt[0])), int(round(all_keypoints[j].pt[1])))
        cv2.line(vis, pt1, pt2, green, 1, lineType=cv2.LINE_AA)

    for kp in all_keypoints:
        c = (int(round(kp.pt[0])), int(round(kp.pt[1])))
        cv2.circle(vis, c, 2, red, thickness=-1, lineType=cv2.LINE_AA)

    stats = {
        "Block grid rows": float(grid_rows),
        "Block grid cols": float(grid_cols),
        "Keypoints (all blocks)": float(len(all_keypoints)),
        "Match pairs drawn": float(len(ordered)),
        "Min point separation (px)": float(min_pixel_dist),
    }
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), stats