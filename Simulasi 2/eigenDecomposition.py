import numpy as np
from sklearn.linear_model import LinearRegression

class CovarianceMatrix:
    def __init__(self, D):
        self.D = D
        self.C = np.matmul(D.T, D)
        self.eigenvalues = None
        self.eigenvectors = None
    def getEigenvalues(self):
        return self.eigenvalues
    def getEigenvectors(self):
        return self.eigenvectors
    def setEigenvalues(self, eigenvalues):
        assert np.iscomplex(eigenvalues).any() == False, "Eigenvalues must be real"
        self.eigenvalues = eigenvalues
    def setEigenvectors(self, eigenvectors):
        assert np.iscomplex(eigenvectors).any() == False, "Eigenvectors must be real"
        self.eigenvectors = eigenvectors
    def calculateEigen(self):
        eig_vals, eig_vecs = np.linalg.eig(self.C)
        if np.iscomplex(eig_vals).any():
            complex_idx = np.where(np.iscomplex(eig_vals))[0]
            eig_vals = np.delete(eig_vals, complex_idx)
            eig_vecs = np.delete(eig_vecs, complex_idx, axis=1)
            eig_vals, eig_vecs = np.real(eig_vals), np.real(eig_vecs)
        self.eigenvalues, self.eigenvectors = eig_vals, eig_vecs
    def setMinEigenVal(self, val):
        assert isinstance(val, (int, float)), "Eigenvalue must be a number"
        assert val >= 0, "Eigenvalue must be positive"
        self.eigenvalues[self.eigenvalues < val] = 0.
        self.eigenvectors[:, self.eigenvalues < val] = 0.

class EigenDecomposition(CovarianceMatrix):
    def __init__(self, D):
        super().__init__(D)
        self.calculateEigen()
    def getPureEigenspectra(self, epsilon=1e-8):
        return self.eigenvectors[:,np.where(self.eigenvalues > epsilon)].reshape(self.D.shape[1], -1)
    def getAbstractEigenspectra(self):
        return np.matmul(self.D, self.getPureEigenspectra())
    def getRatio(self):
        A = self.getAbstractEigenspectra()
        return A[:, 0] / A[:, 1]
    def getProduct(self):
        A = self.getAbstractEigenspectra()
        return A[:, 0] * A[:, 1]    
    def getRegressionLine(self):
        E_acc = self.getPureEigenspectra()
        reg = LinearRegression().fit(E_acc[:, 0].reshape(-1,1), E_acc[:, 1])
        return reg.coef_[0], reg.intercept_
    def getLimit(self):
        ratio = self.getRatio()
        product = self.getProduct()
        minProductIdx, maxProductIdx = np.argmin(product), np.argmax(product)
        minLimit = lambda E_1: -E_1 * ratio[minProductIdx]
        maxLimit = lambda E_1: -E_1 * ratio[maxProductIdx]
        return minLimit, maxLimit
    def getTransformationMatrix(self):
        regCoef, regIntercept = self.getRegressionLine()
        minLimitCoef, maxLimitCoef = self.getLimit()[0](1), self.getLimit()[1](1)
        A_T_1 = np.array([
            [-regCoef, 1],
            [-minLimitCoef, 1]
            ])
        b_T_1 = np.array([regIntercept, 0])
        A_T_2 = np.array([
            [-regCoef, 1],
            [-maxLimitCoef, 1]
            ])
        b_T_2 = np.array([regIntercept, 0])
        T_1 = np.linalg.solve(A_T_1, b_T_1)
        T_2 = np.linalg.solve(A_T_2, b_T_2)
        return np.concatenate((T_1.reshape(2, 1), T_2.reshape(2, 1)), axis=1)
    def getPureSpectra(self):
        return np.matmul(self.getAbstractEigenspectra(), self.getTransformationMatrix())
    def getPureConcentration(self):
        E_acc = self.getPureEigenspectra()
        return np.matmul(np.linalg.inv(self.getTransformationMatrix()), E_acc.T)

if __name__ == '__main__':
    D = np.array([
        [0.26, 0.22, 0.14],
        [0.20, 0.40, 0.80],
        [1.60, 1.20, 0.40],
        [0.12, 0.14, 0.18]
    ])
    eigen = EigenDecomposition(D)
    print("Eigenvalues:")
    print(eigen.getEigenvalues())
    print("Eigenvectors:")
    eigenvector = eigen.getEigenvectors()
    print(eigenvector)
    eigenvector[:, 0] = -eigenvector[:, 0]
    print(eigen.setEigenvectors(eigenvector))
    print("Pure Eigenspectra:")
    print(eigen.getPureEigenspectra())
    print("Abstract Eigenspectra:")
    print(eigen.getAbstractEigenspectra())
    print("Ratio:")
    print(eigen.getRatio())
    print("Product:")
    print(eigen.getProduct())
    print("Regression Line:")
    print(eigen.getRegressionLine())
    print("Limit:")
    print(eigen.getLimit()[0](1), eigen.getLimit()[1](1))
    print("Transformation Matrix:")
    print(eigen.getTransformationMatrix())
    print("Pure Spectra:")
    print(eigen.getPureSpectra())
    print("Pure Concentration:")
    print(eigen.getPureConcentration())