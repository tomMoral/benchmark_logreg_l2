from benchopt import BaseDataset, safe_import_context


with safe_import_context() as import_ctx:
    # Dependencies of download_libsvm are scikit-learn, download and tqdm
    from libsvmdata import fetch_libsvm


class Dataset(BaseDataset):
    name = "news20"
    is_sparse = True

    install_cmd = "conda"
    requirements = ["pip:libsvmdata"]

    parameters = {
        'scaled': [True, False]
    }

    def get_data(self):

        X, y = fetch_libsvm("news20.binary")

        if self.scaled:
            # column scaling
            X -= X.mean(axis=0)
            X /= X.std(axis=0)

        return dict(X=X, y=y)
