from pytorch_metric_learning.utils import accuracy_calculator


class Calculator(accuracy_calculator.AccuracyCalculator):
    def calculate_precision_at_3(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 3,
                                                  self.avg_of_avgs,
                                                  self.return_per_class,
                                                  self.label_comparison_fn)

    def calculate_precision_at_5(self, knn_labels, query_labels, **kwargs):
        return accuracy_calculator.precision_at_k(knn_labels, query_labels[:, None], 5,
                                                  self.avg_of_avgs,
                                                  self.return_per_class,
                                                  self.label_comparison_fn)

    def requires_clustering(self):
        return super().requires_clustering() + ["precision_at_3", "precision_at_5"]

    def requires_knn(self):
        return super().requires_knn() + ["precision_at_3", "precision_at_5"]