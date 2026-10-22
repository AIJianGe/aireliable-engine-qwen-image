from PIL import Image
import numpy as np
import os
paths = {
    'T2I':  r'D:\myProInstallDir\AIReliable\output\image\flux_1787620514_d630.png',
    'EDIT': r'D:\myProInstallDir\AIReliable\output\image\flux_1787620921_1a90.png',
    'I2I':  r'D:\myProInstallDir\AIReliable\output\image\flux_1787621189_8410.png',
}
for name, path in paths.items():
    if not os.path.exists(path):
        print('  [%s] MISSING %s' % (name, path))
        continue
    arr = np.array(Image.open(path).convert('RGB'))
    black = (arr < 5).all()
    h, w, _ = arr.shape
    qs = [
        arr[:h//2, :w//2].mean(),
        arr[:h//2, w//2:].mean(),
        arr[h//2:, :w//2].mean(),
        arr[h//2:, w//2:].mean(),
    ]
    rng = max(qs) - min(qs)
    status = 'OK 非黑有内容' if (not black and rng > 5) else 'FAIL'
    print('  [%-4s] %dx%d mean=%5.1f BLACK=%5s quad_rng=%5.1f -> %s' % (
        name, w, h, float(arr.mean()), str(black), float(rng), status,
    ))
