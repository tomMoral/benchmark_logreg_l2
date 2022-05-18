from benchopt import BaseObjective, safe_import_context

with safe_import_context() as ctx:
    import numpy as np
    from scipy import sparse


def _compute_loss(X, y, lmbd, beta, fit_intercept):
    c = 0
    if fit_intercept:
        beta, c = beta[:-1], beta[-1]
    y_X_beta = y * (X.dot(beta.flatten()) + c)
    l2 = 0.5 * np.dot(beta, beta)

    # This should be more stable than log1p as we don't need to take the exp
    loss = np.logaddexp(0, -y_X_beta).sum() + lmbd * l2
    error = (np.sign(y_X_beta) <= 0).mean()
    return loss, error


class Objective(BaseObjective):
    name = "L2 Logistic Regression"

    parameters = {
        'lmbd': [0.1, 1.0],
        'fit_intercept': [True, False]
    }

    def set_data(self, X, y, X_test=None, y_test=None):
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.lmbd_scale = self._compute_lipschitz_constant() / 4
        msg = "Logistic loss is implemented with y in [-1, 1]"
        assert set(self.y) == {-1, 1}, msg

    def get_one_solution(self):
        return np.zeros(self.X.shape[1] + self.fit_intercept)

    def compute(self, beta):
        train_loss, train_error = _compute_loss(
            self.X, self.y, self.lmbd, beta, self.fit_intercept
        )
        test_loss = test_error = None
        if self.X_test is not None:
            test_loss, test_error = _compute_loss(
                self.X_test, self.y_test, self.lmbd, beta, self.fit_intercept
            )
        return {
            "Train loss": train_loss,
            "Train error": train_error,
            "Test loss": test_loss,
            "Test error": test_error,
            "value": train_loss
        }

    def to_dict(self):
        return dict(
            X=self.X, y=self.y, lmbd=self.lmbd,  # * self.lmbd_scale,
            fit_intercept=self.fit_intercept
        )

    def _compute_lipschitz_constant(self):
        if not sparse.issparse(self.X):
            L = np.linalg.norm(self.X, ord=2) ** 2
        else:
            L = sparse.linalg.svds(self.X, k=1)[1][0] ** 2
        return L
