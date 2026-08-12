import numpy as np

def compute_distance(q, db, metric='L2'):
    if metric.upper() == 'L1':
        return np.sum(np.abs(db - q), axis=-1)
    else:
        return np.sqrt(np.sum((db - q) ** 2, axis=-1))



def retrieve_combined(q_color, q_lbp, q_glcm, db_color, db_lbp, db_glcm, 
                      w_color=1.0, w_lbp=1.0, w_glcm=1.0, metric='L2', top_k=10):
    d_color = compute_distance(q_color, db_color, metric)
    d_lbp = compute_distance(q_lbp, db_lbp, metric)
    d_glcm = compute_distance(q_glcm, db_glcm, metric)
    
    total_w = w_color + w_lbp + w_glcm
    if total_w > 0:
        w_c = w_color / total_w
        w_l = w_lbp / total_w
        w_g = w_glcm / total_w
    else:
        w_c, w_l, w_g = 0.333, 0.333, 0.333
        
    combined_dist = w_c * d_color + w_l * d_lbp + w_g * d_glcm
    
    indices = np.argsort(combined_dist)[:top_k]
    
    return indices, combined_dist[indices], d_color[indices], d_lbp[indices], d_glcm[indices]
