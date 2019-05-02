import numpy as np
import matplotlib.pyplot as plt

def p_x_given_y(y, mus, sigmas):
    cor = sigmas[0,1]
    mu = mus[0] + (cor * (y - mus[1]))
    sigma = 1 - (cor**2)
    return np.random.normal(mu, sigma)

def p_y_given_x(x, mus, sigmas):
    cor = sigmas[0,1]
    mu = mus[1] + (cor * (x - mus[0]))
    sigma = 1 - (cor**2)
    return np.random.normal(mu, sigma)

def gibbs_sampling(mus, sigma, iter=1000):
    samples = np.zeros((iter*2, 2))
    x = 0
    y = np.random.rand() * 10
    samples[0, 0] = x
    samples[0, 1] = y
    for i in range(iter):
        x = p_x_given_y(y, mus, sigma)
        prev_y = y
        y = p_y_given_x(x, mus, sigma)
        samples = np.append(samples, [[x, prev_y]],axis=0)
        samples = np.append(samples, [[x, y]],axis=0)
    return samples

if __name__ == '__main__':
    mus = np.array([1.0, 1.0])
    sigmas = np.array([[1.0, 0.5], [0.5, 1.0]])
    samples = gibbs_sampling(mus, sigmas)
    plt.plot(samples[:, 0], samples[:, 1])
    plt.show()