import numpy as np
import pandas as pd

class FeatureSelector:
    """
    Elimina features con alta correlación para reducir sobreajuste y ruido.
    """
    def __init__(self, data: pd.DataFrame, features: list):
        self.data = data
        self.features = features

    def remove_highly_correlated(self, threshold=0.9):
        """
        Retorna (selected_features, dropped_features).
        - selected_features: las que se mantienen
        - dropped_features: las que se eliminan por correlación alta
        """
        corr_matrix = self.data[self.features].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        selected = [f for f in self.features if f not in to_drop]
        return selected, to_drop
