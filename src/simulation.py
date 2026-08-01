import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import norm

class Simulation:
    """
    Simulates tabular data using a Gaussian Copula generative process via PyMC.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.ecdfs: dict[str, np.ndarray] = {}
        self.columns: list[str] = []
        self.covariance_matrix: np.ndarray | None = None

    def _transform(self, series: pd.Series, col_name: str) -> np.ndarray:
        """
        Applies Probability Integral Transform to map data to a latent normal space.
        """
        sorted_vals = np.sort(series.dropna().values)
        self.ecdfs[col_name] = sorted_vals

        # Calculate empirical CDF (ranks) and scale to [epsilon, 1-epsilon] to avoid inf
        ranks = series.rank(method='average').values
        u = ranks / (len(series) + 1)
        u = np.clip(u, self.epsilon, 1 - self.epsilon)
        
        return norm.ppf(u)

    def _inverse_transform(self, latent_array: np.ndarray, col_name: str) -> np.ndarray:
        """
        Inverts latent normal data back to the original mixed-type feature space.
        """
        u = norm.cdf(latent_array)
        sorted_vals = self.ecdfs[col_name]

        return np.quantile(sorted_vals, u, method='inverted_cdf')

    def fit(self, df: pd.DataFrame, jitter: float = 1e-5) -> None:
        """
        Fits the Gaussian Copula model and enforces positive definiteness.
        """
        self.columns = df.columns.tolist()
        latent_data = np.column_stack([self._transform(df[col], col) for col in self.columns])

        # Compute the 82x82 covariance matrix
        cov_matrix = np.cov(latent_data, rowvar=False)
        
        # Apply Tikhonov regularization (jitter) to the diagonal to ensure Positive Definiteness
        self.covariance_matrix = cov_matrix + np.eye(len(self.columns)) * jitter

    def simulate(self, n_samples: int) -> pd.DataFrame:
        """
        Executes PyMC forward simulation and reconstructs the original feature space.
        """
        if self.covariance_matrix is None:
            raise ValueError("Model must be fitted before simulation.")

        with pm.Model():
            # Define a single 82-dimensional multivariate normal
            latent_space = pm.MvNormal(
                'latent_space',
                mu=np.zeros(len(self.columns)),
                cov=self.covariance_matrix
            )
            
        # Let pm.draw handle the N-sample batch generation.
        # This rigorously guarantees an output shape of (n_samples, 82).
        simulated_latent = pm.draw(latent_space, draws=n_samples)

        simulated_data = {
            col: self._inverse_transform(simulated_latent[:, i], col)
            for i, col in enumerate(self.columns)
        }
        
        return pd.DataFrame(simulated_data)

if __name__ == "__main__":
    from train import load_processed
    
    X_train, X_test, y_train, y_test = load_processed()
    
    df_train = X_train.copy()
    target_col_name = getattr(y_train, 'name', 'target_dropout')
    df_train[target_col_name] = y_train

    # Fit generative model purely on training distribution
    simulator = Simulation()
    simulator.fit(df_train)
    
    # Generate synthetic data
    simulated_full_data = simulator.simulate(n_samples=30000)
    
    # 1. Apply tracking flags
    df_train['Simulated'] = 0
    simulated_full_data['Simulated'] = 1
    
    # 2. Append synthetic data to real training data
    augmented_train = pd.concat([df_train, simulated_full_data], axis=0, ignore_index=True)
    
    # 3. Separate features, targets, and metadata
    y_train_augmented = augmented_train[target_col_name]
    is_simulated_mask = augmented_train['Simulated']
    
    # 4. Drop metadata and targets from the feature matrix
    X_train_augmented = augmented_train.drop(columns=[target_col_name, 'Simulated'])
    
    print(f"Original Training Shape: {X_train.shape}")
    print(f"Augmented Training Shape: {X_train_augmented.shape}")

    X_train_augmented.to_csv('data/processed/X_synthetic.csv', index=False)
    y_train_augmented.to_csv('data/processed/y_synthetic.csv', index=False)
    is_simulated_mask.to_csv('data/processed/simulated_mask.csv', index=False)