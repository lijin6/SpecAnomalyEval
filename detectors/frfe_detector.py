import numpy as np
from scipy.fftpack import fft, ifft
from sklearn.preprocessing import StandardScaler

class FRFEDetector:
    def __init__(self, alpha_range=np.linspace(0.1, 2, 20)):
        self.alpha_range = alpha_range
    
    def frft(self, x, a):
        """Fractional Fourier Transform"""
        N = len(x)
        shft = np.fmod(np.arange(N) + N//2, N).astype(int)
        x_pad = np.hstack([x*np.exp(-1j*np.pi/N*np.arange(N)*np.arange(N))*a, np.zeros(N)])
        y = ifft(fft(x_pad)[:N])
        return y[shft] * np.exp(-1j*np.pi*a*np.arange(N)*np.arange(N)/N)
    
    def find_optimal_order(self, hsi):
        """Determine optimal fractional order"""
        rows, cols, bands = hsi.shape
        max_entropy = -np.inf
        best_order = 0
        
        for alpha in self.alpha_range:
            total_entropy = 0
            for i in range(rows):
                for j in range(cols):
                    spectrum = np.abs(self.frft(hsi[i,j,:], alpha))**2
                    spectrum = spectrum / (spectrum.sum() + 1e-10)
                    entropy = -np.sum(spectrum * np.log(spectrum + 1e-10))
                    total_entropy += entropy
            
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                best_order = alpha
        
        return best_order
    
    def detect(self, hsi):
        import time
        start = time.time()
        """Main detection algorithm"""
        hsi_normalized = hsi / np.max(hsi)
        rows, cols, bands = hsi_normalized.shape
        optimal_order = self.find_optimal_order(hsi_normalized)
        
        # Apply FrFT to all pixels
        frft_result = np.zeros_like(hsi_normalized)
        for i in range(rows):
            for j in range(cols):
                frft_result[i,j,:] = np.abs(self.frft(hsi_normalized[i,j,:], optimal_order))
        
        # Standardize and apply RX detector
        scaler = StandardScaler()
        X = frft_result.reshape(-1, bands)
        X_scaled = scaler.fit_transform(X)
        X_scaled = X_scaled.reshape(rows, cols, bands)
        
        # Global RX detection
        X_flat = X_scaled.reshape(-1, bands)
        mu = np.mean(X_flat, axis=0)
        cov = np.cov(X_flat.T)
        inv_cov = np.linalg.pinv(cov)
        
        scores = np.zeros(rows*cols)
        for i in range(rows*cols):
            x = X_flat[i,:] - mu
            scores[i] = x.T @ inv_cov @ x
            
        processtime = time.time() - start
        print(f"FRFE检测完成，耗时: {processtime:.2f}秒")
        
        return scores.reshape(rows, cols)