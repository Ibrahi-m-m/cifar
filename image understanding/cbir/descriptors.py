import cv2
import numpy as np

def color_histogram(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    
    hist_vec = hist.flatten()
    
    norm = np.linalg.norm(hist_vec)
    if norm > 1e-8:
        hist_vec = hist_vec / norm
        
    return hist_vec

def compute_lbp(gray_image):
    padded = np.pad(gray_image, 1, mode='edge')
    
    n0 = padded[:-2, :-2]
    n1 = padded[:-2, 1:-1]
    n2 = padded[:-2, 2:]
    n3 = padded[1:-1, 2:]
    n4 = padded[2:, 2:]
    n5 = padded[2:, 1:-1]
    n6 = padded[2:, :-2]
    n7 = padded[1:-1, :-2]
    
    center = gray_image
    
    lbp = ((n0 >= center).astype(np.uint8) << 0) | \
          ((n1 >= center).astype(np.uint8) << 1) | \
          ((n2 >= center).astype(np.uint8) << 2) | \
          ((n3 >= center).astype(np.uint8) << 3) | \
          ((n4 >= center).astype(np.uint8) << 4) | \
          ((n5 >= center).astype(np.uint8) << 5) | \
          ((n6 >= center).astype(np.uint8) << 6) | \
          ((n7 >= center).astype(np.uint8) << 7)
          
    return lbp

def lbp_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lbp = compute_lbp(gray)
    
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    
    hist_vec = hist.astype(np.float32)
    norm = np.linalg.norm(hist_vec)
    if norm > 1e-8:
        hist_vec = hist_vec / norm
        
    return hist_vec

def get_glcm_features(img, num_levels=8):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    divisor = 256 // num_levels
    quantized = (gray // divisor).astype(np.int32)
    quantized = np.clip(quantized, 0, num_levels - 1)
    
    offsets = [(0, 1), (-1, 1), (1, 0), (1, 1)]
    
    features = []
    
    for dy, dx in offsets:
        glcm = np.zeros((num_levels, num_levels), dtype=np.float32)
        
        if dy == 0:
            img1 = quantized[:, :-dx]
            img2 = quantized[:, dx:]
        elif dx == 0:
            img1 = quantized[:-dy, :]
            img2 = quantized[dy:, :]
        elif dy < 0:
            img1 = quantized[-dy:, :-dx]
            img2 = quantized[:dy, dx:]
        else:
            img1 = quantized[:-dy, :-dx]
            img2 = quantized[dy:, dx:]
            
        np.add.at(glcm, (img1, img2), 1)
        
        glcm = glcm + glcm.T
        
        s = glcm.sum()
        if s > 0:
            glcm /= s
            
        i, j = np.meshgrid(np.arange(num_levels), np.arange(num_levels), indexing='ij')
        
        contrast = np.sum(((i - j) ** 2) * glcm)
        
        energy = np.sum(glcm ** 2)
        
        homogeneity = np.sum(glcm / (1.0 + (i - j) ** 2))
        
        p_nonzero = glcm[glcm > 0]
        entropy = -np.sum(p_nonzero * np.log2(p_nonzero)) if len(p_nonzero) > 0 else 0.0
        
        p_i = np.sum(glcm, axis=1)
        p_j = np.sum(glcm, axis=0)
        
        mu_i = np.sum(np.arange(num_levels) * p_i)
        mu_j = np.sum(np.arange(num_levels) * p_j)
        
        var_i = np.sum(((np.arange(num_levels) - mu_i) ** 2) * p_i)
        var_j = np.sum(((np.arange(num_levels) - mu_j) ** 2) * p_j)
        
        std_i = np.sqrt(var_i)
        std_j = np.sqrt(var_j)
        
        if std_i * std_j > 1e-8:
            correlation = np.sum((i - mu_i) * (j - mu_j) * glcm) / (std_i * std_j)
        else:
            correlation = 0.0
            
        features.extend([contrast, energy, homogeneity, entropy, correlation])
        
    features_vec = np.array(features, dtype=np.float32)
    
    norm = np.linalg.norm(features_vec)
    if norm > 1e-8:
        features_vec = features_vec / norm
        
    return features_vec
