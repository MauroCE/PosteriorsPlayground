from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import nn, utils, func
import torchopt
import posteriors

dataset = MNIST(root="./data", transform=ToTensor(), download=True)
train_loader = utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
num_data = len(dataset)

classifier = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10))
params = dict(classifier.named_parameters())


def log_posterior(params, batch):
    images, labels = batch
    images = images.view(images.size(0), -1)
    output = func.functional_call(classifier, params, images)
    log_post_val = (
        -nn.functional.cross_entropy(output, labels)
        + posteriors.diag_normal_log_prob(params) / num_data
    )
    return log_post_val, output


transform = posteriors.vi.diag.build(
    log_posterior, torchopt.adam(), temperature=1 / num_data
)

state = transform.init(params)

for batch in train_loader:
    state = transform.update(state, batch)
