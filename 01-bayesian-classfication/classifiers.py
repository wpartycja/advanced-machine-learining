import numpy as np
from sklearn.preprocessing import StandardScaler


class NaiveBayes:
    name = "Naive Bayes"

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the model using training data.
 
        Args:
            X (np.array): design matrix, containing in each row the values
                of features for a single observation
            y (np.array): vector containing values of binary
                target variable for observations

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, variance and prior probability for each class
        self._mean = np.zeros((n_classes, n_features))
        self._variance = np.zeros((n_classes, n_features))
        self._prior_class = np.zeros(n_classes)

        for idx, cls in enumerate(self._classes):
            X_class = X[y == cls]  # only rows of this class
            self._mean[idx, :] = X_class.mean(axis=0)
            self._variance[idx, :] = X_class.var(axis=0)
            self._prior_class[idx] = X_class.shape[0] / n_samples

    def predict_proba(self, X_test: np.array) -> list:
        """
        Computes predicted posterior probabilities for classes.

        Args:
            X_test (np.array): matrix in which rows are feature values for observations

        Returns:
            prob (list): predicted posterior probabilities
        """

        prob = []
        # calculate posterior probability for each class
        for x in X_test:
            posteriors = []
            for idx, cls in enumerate(self._classes):
                prior = np.log(self._prior_class[idx])
                posterior = np.sum(np.log(self._pdf(idx, x)))
                posterior = posterior + prior
                posteriors.append(posterior)
            prob.append(posteriors)
  
        return prob

    def predict(self, X_test: np.array) -> list:
        """
        Assigns the predicted class (0 or 1) for observations.

        Args:
            X_test (list): matrix in which rows are feature values for observations
        """
        probs = self.predict_proba(X_test)

        results = []
        for prob in probs:
            results.append(self._classes[np.argmax(prob)])

        return results

    def get_params(self) -> tuple:
        """
        Returns a list containing the estimated parameters.


        Returns:
            params (list): estimated parameters
        """
        return self._mean, self._variance, self._prior_class

    def _pdf(self, class_idx: int, x: np.array) -> float:
        """
        Calculate Probability Density Funciton

        Args:
            class_idx (int): index of a class

        Returns:
            PDF value
        """
        mean = self._mean[class_idx]
        var = self._variance[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


class LDA():
    name = "LDA"

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the model using training data.

        Args:
            X (np.array): design matrix, containing in each row the values
                of features for a single observation
            y (np.array): vector containing values of binary
                target variable for observations

        Returns:
            None
        """
        self._classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self._classes)
        
        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self._mean = np.zeros((n_classes, n_features))
        self._cov_matrices = []
       
        # caluclate mean and covariance matrices      
        for idx, cls in enumerate(self._classes):
            X_class = X_scaled[y == cls]  # only rows of this class
            self._mean[idx, :] = X_class.mean(axis=0)
            self._cov_matrices.insert(idx, self.calculate_covariance_matrix(X_class))
            
        mean_diff = np.atleast_1d(self._mean[0] - self._mean[1])
        total_covariance = self._cov_matrices[0] + self._cov_matrices[1]
        
        self.vector = np.linalg.inv(total_covariance).dot(mean_diff)
    
    def predict_proba(self, X_test: np.array) -> list:
        """
        Computes predicted posterior probabilities for classes.

        Args:
            X_test (np.array): matrix in which rows are feature values for observations

        Returns:
            prob (list): predicted posterior probabilities
        """
        # Standardize the features
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        y_prob = []
        for sample in X_test_scaled:
            h = sample.dot(self.vector)
            y_prob.append(h)
        return y_prob
        
    def predict(self, X_test: np.array) -> np.array:
        """
        Assigns the predicted class (0 or 1) for observations.

        Args:
            X_test (list): matrix in which rows are feature values for observations
        """
        probabilities = self.predict_proba(X_test)
        
        y_pred = []
        for prob in probabilities:
            y_pred.append(1 * (prob < 0))
        return y_pred
    
    def get_params(self) -> tuple:
        """
        Returns a list containing the estimated parameters.
        
        
        Returns:
            params (list): estimated parameters
        """
        return self._mean, self._cov_matrices
                
    @staticmethod
    def calculate_covariance_matrix(X: np.array) -> np.array:
        """ 
        Calculate the covariance matrix for the dataset X 
        
        Args:
            X (np.array): array with features
            
        Return:
            Covariance matrix (np.array)
        """
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 2)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))

        return np.array(covariance_matrix, dtype=float)

    
class QDA():
    name = "QDA"

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fits the model using training data.
        
        Args:
            X (np.array): design matrix, containing in each row the values
                of features for a single observation
            y (np.array): vector containing values of binary
                target variable for observations
        
        Returns:
            None
        """
        self._classes = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self._classes)
        
        self._mean = np.zeros((n_classes, n_features))
        self._cov_matrices = []
       
        # calculate mean and covariance matrices      
        for idx, cls in enumerate(self._classes):
            X_class = X[y == cls]  # only rows of this class
            self._mean[idx, :] = X_class.mean(axis=0)
            self._cov_matrices.insert(idx, self.calculate_covariance_matrix(X_class))
              
    def predict_proba(self, X_test: np.array) -> list:
        """
        Computes predicted posterior probabilities for classes.

        Args:
            X_test (np.array): matrix in which rows are feature values for observations

        Returns:
            prob (list): predicted posterior probabilities
        """
        probs = []
        for sample in X_test:
            discriminant_vals = []
            for idx, cls in enumerate(self._classes):
                mean_diff = np.atleast_1d(sample - self._mean[idx])
                covariance_inv = np.linalg.inv(self._cov_matrices[idx])
                discriminant_val = -0.5 * mean_diff.dot(covariance_inv).dot(mean_diff.T) - 0.5 * np.log(np.linalg.det(self._cov_matrices[idx]))
                discriminant_vals.append(discriminant_val)
            probs.append(discriminant_vals)
        return probs
        
    def predict(self, X_test: np.array) -> np.array:
        """
        Assigns the predicted class (0 or 1) for observations.

        Args:
            X_test (list): matrix in which rows are feature values for observations
        """
        probs = self.predict_proba(X_test)
        y_pred = []
        for discriminant_val in probs:
            predicted_class = np.argmax(discriminant_val)
            y_pred.append(self._classes[predicted_class])
        return y_pred
    
    def get_params(self) -> tuple:
        """
        Returns a list containing the estimated parameters.
        
        
        Returns:
            params (list): estimated parameters
        """
        return self._mean, self._cov_matrices


    @staticmethod
    def calculate_covariance_matrix(X: np.array) -> np.array:
        """ 
        Calculate the covariance matrix for the dataset X 
        
        Args:
            X (np.array): array with features
            
        Return:
            Covariance matrix (np.array)
        """
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples-1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))

        return np.array(covariance_matrix, dtype=float)
    