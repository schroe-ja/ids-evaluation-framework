import logging
import unittest

from types import SimpleNamespace

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist

from ids_eval.enumeration.internal_label import InternalLabel
from ids_eval.dataset_pipeline.dataset_feature_selector import DatasetFeatureSelector

class TestFeatureSelection(unittest.TestCase):
    def setUp(self) -> None:
        self.c = 1.0
        self.penalty = SimpleNamespace(value="l2")
        self.solver = SimpleNamespace(value="lbfgs")
        self.max_iter = 200
        self.threshold = 0.01

    @staticmethod
    def _load_test_data() -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        list[str],
        str,
    ]:
        (images_train, labels_train), (images_test, labels_test) = mnist.load_data()

        # Test only a part of the dataset
        train_images = images_train[:200].reshape(200, -1).astype(np.float32)
        test_images = images_test[:50].reshape(50, -1).astype(np.float32)

        feature_names = [f"pixel_{i}" for i in range(train_images.shape[1])]
        train_index = range(200)
        test_index = range(200, 250)
    
        X_train = pd.DataFrame(train_images, columns=feature_names, index=train_index)
        X_test = pd.DataFrame(test_images, columns=feature_names, index=test_index)

        target_col = InternalLabel.TARGET_LABEL.value
        y_train = pd.DataFrame({target_col: labels_train[:200]}, index=train_index)
        y_test = pd.DataFrame({target_col: labels_test[:50]}, index=test_index)

        return X_train, X_test, y_train, y_test, feature_names, target_col

    def test_logistic_regression(self) -> None:
        
        X_train, X_test, y_train, y_test, feature_names, target_col = self._load_test_data()

        params = SimpleNamespace(
            C=self.c,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            threshold=self.threshold
        )
        config = SimpleNamespace(
            data_manager=SimpleNamespace(
                dataset=[
                    SimpleNamespace(
                        feature_selector=SimpleNamespace(params=params)
                    )
                ]
            ),
            general=SimpleNamespace(seed=42)
        )

        selector_instance = DatasetFeatureSelector.__new__(DatasetFeatureSelector)
        selector_instance.config = config
        selector_instance.logger = logging.getLogger("test")
        selector_instance.metadata = {"steps": []}

        result = selector_instance._logistic_regression(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            config_index=0,
        )

        # Check output shape, labels, and that the selection metadata was recorded.
        self.assertEqual(len(result), len(X_train) + len(X_test))
        self.assertIn(target_col, result.columns)
        self.assertTrue(result.index.is_unique)

        expected_target = pd.concat([y_train, y_test])[target_col].reindex(result.index)
        pd.testing.assert_series_equal(result[target_col], expected_target)

        step = selector_instance.metadata["steps"][-1]
        self.assertEqual(step["action"], "logistic_regression_feature_selection")
        self.assertEqual(step["selected_count"], len(step["selected_features"]))
        self.assertTrue(set(step["selected_features"]).issubset(feature_names))
        self.assertTrue(set(step["selected_features"]).issubset(result.columns))

