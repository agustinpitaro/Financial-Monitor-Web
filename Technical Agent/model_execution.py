from sklearn.metrics import accuracy_score

class ModelExecutor:
    """
    Clase para entrenar y evaluar el modelo, 
    incluyendo un método de validación walk-forward.
    """
    def __init__(self, model):
        self.model = model

    def train(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        return self.model.score(X, y)

    def walk_forward_validation(self, X, y, initial_train_size, test_size):
        """
        Divide X e y en ventanas que avanzan ('walk-forward').
        En cada bloque, entrena y evalúa.
        Devuelve los accuracies en cada iteración.
        """
        n_samples = len(X)
        scores = []
        start = 0
        while (start + initial_train_size + test_size) <= n_samples:
            train_end = start + initial_train_size
            test_end  = train_end + test_size

            X_train_wf = X.iloc[start:train_end]
            y_train_wf = y.iloc[start:train_end]
            X_test_wf  = X.iloc[train_end:test_end]
            y_test_wf  = y.iloc[train_end:test_end]

            self.model.fit(X_train_wf, y_train_wf)
            y_pred = self.model.predict(X_test_wf)
            score = accuracy_score(y_test_wf, y_pred)
            scores.append(score)

            start += test_size
        return scores
