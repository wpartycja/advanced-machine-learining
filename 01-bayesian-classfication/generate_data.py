import numpy as np

def generate_data_scheme_1(n, mean):
    # Generate binary variable from Bernoulli distribution
    y = np.random.binomial(1, 0.5, n)
    
    # Generate features for class 0
    class_0_features = np.random.normal(0, 1, (n, 2))
    
    # Generate features for class 1
    class_1_features = np.random.normal(mean, 1, (n, 2))
    
    # Combine features based on binary variable
    X = np.where(y[:, np.newaxis] == 1, class_1_features, class_0_features)
    return X, y


def generate_data_scheme_2(n, mean, rho):
    # Generate binary variable from Bernoulli distribution
    y = np.random.binomial(1, 0.5, n)

    # Generate features for class 0
    mean_class_0 = np.array([0, 0])
    cov_class_0 = np.array([[1, rho], [rho, 1]])
    class_0_features = np.random.multivariate_normal(mean_class_0, cov_class_0, n)

    # Generate features for class 1
    mean_class_1 = np.array([mean, mean])
    cov_class_1 = np.array([[1, rho], [rho, 1]])
    class_1_features = np.random.multivariate_normal(mean_class_1, cov_class_1, n)

    # Concatenate features based on binary variable
    X = np.where(y[:, np.newaxis] == 1, class_1_features, class_0_features)

    return X, y