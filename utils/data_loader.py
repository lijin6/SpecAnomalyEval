import scipy.io as sio
import numpy as np

def load_mat_file(file_path: str) -> dict:
    """智能加载.mat文件"""
    def find_3d_array(data):
        for key in data:
            if isinstance(data[key], np.ndarray) and data[key].ndim == 3:
                return data[key]
        return None

    mat_data = sio.loadmat(file_path)
    data = find_3d_array(mat_data)
    gt = mat_data.get('gt', mat_data.get('map', None))
    
    if data is not None and np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32)
    
    return {'data': data, 'gt': gt}