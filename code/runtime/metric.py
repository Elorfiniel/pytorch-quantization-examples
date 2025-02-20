from mmengine.evaluator import BaseMetric


class Accuracy(BaseMetric):
  def process(self, data_batch, data_samples):
    probs, labels = data_samples
    self.results.append(dict(
      n_correct=(probs.argmax(dim=1) == labels).sum().cpu(),
      n_samples=len(labels),
    ))

  def compute_metrics(self, results):
    n_correct = sum([x['n_correct'] for x in results])
    n_samples = sum([x['n_samples'] for x in results])
    return dict(accuracy=100.0 * n_correct / n_samples)
