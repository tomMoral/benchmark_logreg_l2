from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from tick.linear_model import ModelLogReg
    from tick.prox import ProxL2Sq
    from tick.solver import SVRG


class Solver(BaseSolver):
    name = 'svrg-tick'

    install_cmd = 'conda'
    requirements = [
        'pip:git+https://github.com/X-DataInitiative/tick.git'
    ]
    references = [
        'Bacry, Emmanuel, et al. "Tick: a Python library for statistical learning, '
        'with a particular emphasis on time-dependent modelling." '
        'arXiv preprint arXiv:1707.03003 (2017).'
    ]

    parameters = {
        'step_type': [
            'bb',
            'fixed',
        ]
    }

    def set_objective(self, X, y, lmbd):
        self.X, self.y, self.lmbd = X, y, lmbd
        # Reponse vector used for tick logreg must be {1.0, -1.0}
        self.y[y == 0] = -1.0
        self.clf = SVRG(tol=1e-16, verbose=False, step_type=self.step_type)

    def run(self, n_iter):
        model = ModelLogReg(fit_intercept=False).fit(self.X, self.y)
        # Scaled with 1/n_samples as logistic objective is taken with mean
        prox = ProxL2Sq(self.lmbd / self.X.shape[0])
        optimal_step_size = 1 / model.get_lip_max()
        self.clf.max_iter = n_iter
        self.clf.set_model(model).set_prox(prox)
        self.coef_ = self.clf.solve(step=optimal_step_size)
        
    def get_result(self):
        return self.coef_.ravel()
