from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

class ModelTuner:
    """
    Clase para encapsular la lógica de ajuste de hiperparámetros (grid search).
    """
    def __init__(self, param_grid: dict, cv_splits=5):
        self.param_grid = param_grid
        self.cv_splits = cv_splits
        self.best_model = None
        self.best_params = None
        self.best_score  = None

    def tune(self, X_train, y_train):
        """
        Ejecuta GridSearchCV con validación temporal (TimeSeriesSplit)
        y asigna el mejor modelo a self.best_model
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