from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

class ModelTuner:
    """
    Clase que encapsula la búsqueda de hiperparámetros con GridSearchCV
    usando TimeSeriesSplit para datos secuenciales.
    """
    def __init__(self, param_grid, cv_splits=5):
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def tune(self, X_train, y_train):
        """
        Aplica GridSearchCV (RandomForest) con validación temporal.
        Retorna: (best_model, best_params, best_score)
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(random_state=42),
            param_grid=self.param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)

        self.best_model  = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        self.best_score  = grid_search.best_score_
        return self.best_model, self.best_params, self.best_score
