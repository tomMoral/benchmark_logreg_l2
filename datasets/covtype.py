from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    from sklearn.datasets import fetch_covtype
    from sklearn.preprocessing import StandardScaler


class Dataset(BaseDataset):
    name = "covtype_binary"

    install_cmd = 'conda'
    requirements = ['pip:scikit-learn']

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):
        X, y = fetch_covtype(return_X_y=True)
        y[y != 2] = -1
        y[y == 2] = 1  # try to separate class 2 from the other 6 classes.

        if self.scaled:
            # column scaling
            scaler = StandardScaler(with_mean=False)
            X = scaler.fit_transform(X)

        return dict(X=X, y=y)
