import numpy as np

def generate_dataset(n, a):
    # Generate binary variable from Bernoulli distribution
    binary_variable = np.random.binomial(1, 0.5, n)
    
    # Generate features for class 0
    class_0_features = np.random.normal(0, 1, (n, 2))
    
    # Generate features for class 1
    class_1_features = np.random.normal(a, 1, (n, 2))
    
    # Combine features based on binary variable
    features = np.where(binary_variable[:, np.newaxis]==1, class_1_features, class_0_features)
    
    return features, binary_variable

# Parameters
n = 1000  # number of observations
p = 2     # number of features
a = 2     # mean for class 1

# Generate dataset
features, binary_variable = generate_dataset(n, a)

# Print first few observations
print("Features (first 5 observations):\n", features[:5])
print("Binary variable (first 5 observations):\n", binary_variable[:5])
