import numpy as np
from scipy.linalg import inv
import time

class GlobalRXDetector:
    def __init__(self, batch_size=200):
        """
        全局RX异常检测器
        
        参数:
            batch_size: 分批处理时的批大小（针对大图像内存优化）
        """
        self.batch_size = batch_size

    def detect(self, data):
        
        start_time = time.time()
        h, w, c = data.shape
        pixels = h * w
        
        # 1. 数据重塑和统计量计算
        reshaped_data = data.reshape(-1, c).T  # (C, H*W)
        
        # 计算全局统计量
        mean_vec = np.mean(reshaped_data, axis=1, keepdims=True)  # (C, 1)
        cov_mat = np.cov(reshaped_data)  # (C, C)
        inv_cov = inv(cov_mat)  # 协方差矩阵求逆
        
        # 2. 分批处理（针对大图像）
        if pixels > self.batch_size:
            result = np.zeros(pixels)
            num_batches = int(np.ceil(pixels / self.batch_size))
            
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, pixels)
                batch = reshaped_data[:, start:end]  # (C, batch_size)
                
                # 计算RX得分
                diff = batch - mean_vec
                rx_scores = np.sum(diff * (inv_cov @ diff), axis=0)  # (batch_size,)
                result[start:end] = rx_scores
        else:
            # 小图像直接计算
            diff = reshaped_data - mean_vec
            result = np.sum(diff * (inv_cov @ diff), axis=0)
        
        # 3. 结果后处理
        result = result.reshape(h, w)  # 恢复图像形状
        result = np.abs(result)  # 确保非负
        
        # 4. 归一化处理
        if np.max(result) > 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        
        process_time = time.time() - start_time
        print(f"GlobalRX检测完成，耗时: {process_time:.2f}秒")
        
        return result

    def __call__(self, data):
        """支持直接调用实例"""
        return self.detect(data)