import numpy as np
from scipy.linalg import pinv
import time

class UNRSDetector:
    def __init__(self, win_out=5, win_in=3, lambda_=0.1):
        """
        无监督最近正则化子空间异常检测器(UNRS)
        
        参数:
            win_out: 外部窗口尺寸(奇数)
            win_in: 内部窗口尺寸(奇数且小于win_out) 
            lambda_: 正则化参数
        """
        self.win_out = win_out
        self.win_in = win_in
        self.lambda_ = lambda_

    def detect(self, data):
        """
        执行异常检测
        
        参数:
            data: 3D高光谱数据(H, W, C)
            
        返回:
            result: 2D检测结果图(H, W)
        """
        start = time.time()
        h, w, c = data.shape
        result = np.zeros((h, w))
        t = self.win_out // 2
        t1 = self.win_in // 2
        num_sam = self.win_out**2 - self.win_in**2

        # 1. 数据填充（镜像填充）
        padded_data = np.pad(data, 
                           ((t, t), (t, t), (0, 0)),
                           mode='reflect')

        # 2. 滑动窗口处理
        for i in range(t, t + h):
            for j in range(t, t + w):
                # 获取当前窗口
                block = padded_data[i-t:i+t+1, j-t:j+t+1, :]
                
                # 中心像元
                y = padded_data[i, j, :].reshape(1, -1)  # (1, C)
                
                # 创建内窗口掩膜
                mask = np.ones((self.win_out, self.win_out), dtype=bool)
                mask[t-t1:t+t1+1, t-t1:t+t1+1] = False
                
                # 提取背景样本
                H = block[mask].T  # (C, num_sam)
                
                # 3. UNRS核心计算
                try:
                    # 计算差分矩阵
                    z = H - y.T  # (C, num_sam)
                    
                    # 计算Gram矩阵和正则化
                    Gram = z.T @ z  # (num_sam, num_sam)
                    C = Gram + self.lambda_ * np.eye(num_sam)
                    
                    # 计算权重 (公式12)
                    invC = pinv(C)
                    weights = np.sum(invC, axis=1) / np.sum(invC)
                    
                    # 重构误差(L1范数)
                    y_hat = (H @ weights).reshape(1, -1)  # (1, C)
                    result[i-t, j-t] = np.linalg.norm(y - y_hat, ord=1)
                    
                except Exception as e:
                    result[i-t, j-t] = 0

        # 4. 结果后处理
        processtime = time.time() - start
        print(f"UNRS检测完成，耗时: {processtime:.2f}秒")
        result = self._post_process(result)
        return result

    def _post_process(self, result):
        """结果后处理"""
        # 替换负值为0
        result[result < 0] = 0
        # 归一化到[0,1]范围
        if np.max(result) > 0:
            result = (result - np.min(result)) / (np.max(result) - np.min(result))
        return result

    def __call__(self, data):
        """支持直接调用实例"""
        return self.detect(data)