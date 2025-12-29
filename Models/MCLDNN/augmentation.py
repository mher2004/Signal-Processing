import numpy as np

def random_phase_rotation(X):
    # X shape: (N, 2, 128)
    theta = np.random.uniform(0, 2*np.pi, size=(X.shape[0], 1))
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    I = X[:,0,:]
    Q = X[:,1,:]

    I_new = cos_t * I - sin_t * Q
    Q_new = sin_t * I + cos_t * Q

    X[:,0,:] = I_new
    X[:,1,:] = Q_new
    return X

def normalize_iq(X):
    power = np.sqrt(np.mean(X[:,0,:]**2 + X[:,1,:]**2, axis=1, keepdims=True))
    X[:,0,:] /= (power + 1e-8)
    X[:,1,:] /= (power + 1e-8)
    return X

def random_frequency_offset(X, max_offset=0.05):
    N, _, L = X.shape
    t = np.arange(L)

    offsets = np.random.uniform(-max_offset, max_offset, size=(N,1))
    phase = 2 * np.pi * offsets * t

    cos_p = np.cos(phase)
    sin_p = np.sin(phase)

    I = X[:,0,:]
    Q = X[:,1,:]

    X[:,0,:] = cos_p * I - sin_p * Q
    X[:,1,:] = sin_p * I + cos_p * Q
    return X

def awgn_jitter(X, sigma=0.01):
    noise = np.random.normal(0, sigma, X.shape)
    return X + noise

def augment_iq(X):
    X = X.copy()

    if np.random.rand() < 0.8:
        X = random_phase_rotation(X)

    if np.random.rand() < 0.5:
        X = random_frequency_offset(X)

    if np.random.rand() < 0.3:
        X = awgn_jitter(X)

    return X

def data_generator(X, Y, batch_size=256):
    idx = np.arange(len(X))
    while True:
        np.random.shuffle(idx)
        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i+batch_size]
            Xb = augment_iq(X[batch_idx])

            # normalize AFTER augmentation
            Xb = normalize_iq(Xb)

            yield (
                [
                    np.expand_dims(Xb, axis=3),          # (N,2,128,1)
                    np.expand_dims(Xb[:,0,:], axis=2),   # I
                    np.expand_dims(Xb[:,1,:], axis=2)    # Q
                ],
                Y[batch_idx]
            )

