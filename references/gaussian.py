import numpy as np
import time
np.seterr(all = "print")
{'over': 'warn', 'divide': 'warn', 'invalid': 'warn', 'under': 'ignore'}

class gaussian:
    def __init__(self, mu, variance):
        try: 
            int(mu)
            self.dim = 1
        except:
            self.dim = len(mu)
        self.mu = mu if self.dim == 1 else np.array(mu)
        self.variance = variance if self.dim == 1 else np.array(variance)
        
        self.invvar = 1 / variance if self.dim == 1 else np.linalg.inv(self.variance)
        self.coeff = 1 / np.sqrt(2*np.pi*variance) if self.dim == 1 else \
            1 / np.sqrt((2*np.pi)**self.dim * np.linalg.det(self.variance))

    def pdf(self, x):
        x = np.array(x)
        return self.coeff * np.exp(-0.5 * (x - self.mu) ** 2 * self.invvar) if self.dim == 1 else \
            self.coeff * np.exp((-0.5 * (x - self.mu) @ self.invvar @ (x - self.mu)))
    
    def multiply(self, G):
        """return new distribution with the product of two new gaussians in following rules:
            N(m1, v1) * N(m2, v2) = N((m1v2+m2v1)/(v1 + v2), 1/(1/v1 + 1/v2))
            multivariate case: (matrix cookbook)
            cov = inv(inv(cov1) + inv(cov2))
            mu = cov * invcov1 * m1 + cov *invcov2 * m2
            """
        if self.dim == 1:
            mu = (self.mu * G.variance + self.variance * G.mu) / (self.variance + G.variance)
            var =1 / ( 1 / self.variance + 1/ G.variance )
        else:
            var = np.linalg.inv(self.invvar + G.invvar)
            mu = var @ self.invvar @ self.mu + var @ G.invvar @ G.mu
        return gaussian(mu, var)

G1 = gaussian(10, 8)
G2 = gaussian(13, 2)
G3 = G1.multiply(G2)
print(G3.mu, G3.variance)



