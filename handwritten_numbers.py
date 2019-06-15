import numpy as np
from itertools import product, cycle

import torch
import torchvision.transforms as trans
import torchvision.datasets as dataset


# def group_img_by_digit(mnist):
#     ixc_pair = [(mnist[i][1], i) for i in range(len(mnist))]
#     ixc_pair  = sorted(ixc_pair, key=lambda x: x[0])

#     current = ixc_pair[0][0]
#     groups = {current: []}

#     for num, idx in ixc_pair:
#         if num == current:
#             groups[current].append(idx)
#         else:
#             current = num
#             groups[current] = [idx]

#     return groups


def trim_leading_zeros(img_shape, images, targets):
    empty_image = torch.zeros_like(images[0, 0])

    for i in range(images.shape[0]):
        for j in range(images.shape[1] - 1):
            if targets[i * images.shape[1] + j] == 0:
                images[i, j] = empty_image
            else:
                break

def process_inputs(images, batch_shape, digit_shape):
    bs, nd = batch_shape
    images = [img.view(*digit_shape) for img in images]
    images = [torch.cat(images[i:i+nd], dim=1) for i in range(bs)]
    return torch.stack(images).view(*batch_shape, -1)


def process_labels(labels, batch_shape):
    bs, nd = batch_shape
    pof10 = torch.tensor([10 ** (p-1) for p in np.arange(nd,0,-1)])

    labels = torch.as_tensor([labels[i:i+nd] for i in range(bs)])
    labels *= pof10

    return labels.sum(dim=-1)

class HandwrittenNumberGenerator:
    def __init__(self, mnist, ndigits, digit_shape, supervised=True, dataset_size=1000,
            batch_size=32, leading_zeros=False, infinite_gen=True, random_state=None):
        self.mnist = mnist
        self.ndigts = ndigits
        self.digit_shape = digit_shape
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.leading_zeros = leading_zeros
        self.infinite_gen = infinite_gen
        self.supervised = supervised

        if random_state is None or isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)

        self.rng = random_state
        self.init_state = random_state.get_state()

        # store which datapoints belong to which class
        # self._idx_by_digit = group_img_by_digit(mnist)


    def __len__(self):
        return int(np.ceil(self.dataset_size / self.batch_size))

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()
        if not self.infinite_gen:
            self.reset()

    def reset(self):
        self.rng.set_state(self.init_state)

    def next_batch(self):
        batch_shape = self.batch_size, self.ndigts

        idx = self.rng.choice(range(len(self.mnist)), size=batch_shape).reshape(-1)

        inputs, labels = [],[]
        for i in idx:
            img, lbl = self.mnist[i]
            inputs.append(img)
            labels.append(lbl)

        inputs = process_inputs(inputs, batch_shape, self.digit_shape)
        if not self.leading_zeros:
            trim_leading_zeros(self.digit_shape, inputs, labels)

        inputs = inputs.view(self.batch_size, -1)

        if self.supervised:
            targets = process_labels(labels, batch_shape)
        else:
            targets = inputs

        return inputs, targets


class SequenceGenerator:
    def __init__(self, generators):
        self.generators = generators

    def __len__(self):
        return len(self.generators[0])

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

    def next_batch(self):
        inputs, targets = [], []

        for gen in self.generators:
            images, labels = gen.next_batch()
            inputs.append(images)
            targets.append(labels)

        inputs = torch.stack(inputs, dim=1)
        targets = torch.stack(targets, dim=1)

        return inputs, targets


def load_raw(data_path, crop_shape=(22, 20), download=False):
    transform = trans.Compose([
        trans.CenterCrop(crop_shape),
        trans.ToTensor(),
        trans.Lambda(lambda x: x.view(-1, np.prod(x.shape)))
    ])

    train_data = dataset.MNIST(
        root=data_path, train=True, transform=transform, download=download)
    test_data = dataset.MNIST(
        root=data_path, train=False, transform=transform, download=download)

    return train_data, test_data


def load_handwritten_number_data(
        mnist_path,
        dataset_size=1000,
        ndigits=3,
        digit_shape=(22, 20),
        supervised=True,
        batch_size=50,
        leading_zeros=False,
        infinite_gen=True,
        random_state=None,
        download=False
    ):
    train_raw, test_raw = load_raw(mnist_path, digit_shape, download)

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    training_set = HandwrittenNumberGenerator(
        mnist=train_raw,
        ndigits=ndigits,
        digit_shape=digit_shape,
        supervised=supervised,
        dataset_size=dataset_size,
        batch_size=batch_size,
        leading_zeros=False,
        infinite_gen=True,
        random_state=random_state.randint(2**32-1) if not infinite_gen else random_state
    )

    test_set = HandwrittenNumberGenerator(
        mnist=test_raw,
        ndigits=ndigits,
        digit_shape=digit_shape,
        supervised=supervised,
        dataset_size=dataset_size,
        batch_size=batch_size,
        leading_zeros=False,
        infinite_gen=True,
        random_state=random_state.randint(2**32-1) if not infinite_gen else random_state
    )

    validation_set = HandwrittenNumberGenerator(
        mnist=train_raw,
        ndigits=ndigits,
        digit_shape=digit_shape,
        supervised=supervised,
        # validation set size as a fraction of training set size
        dataset_size=int(0.1 * dataset_size),
        batch_size=batch_size,
        leadning_zeros=False,
        infinite_gen=True,
        rng=random_state.randint(2**32-1) if not infinite_gen else random_state
    )

    return training_set, test_set, validation_set


def load_handwritten_sequence(
        mnist_path, size,
        seqlen, ndigits,
        digit_shape,
        supervised=True,
        batch_size=50,
        infinite_gen=True,
        leading_zeros=False,
        train_filters=None,
        test_filters=None,
        random_state=None,
        download=False
    ):

    train_raw, test_raw = load_raw(mnist_path, digit_shape, download)

    training_gen, validation_gen, test_gen = [], [], []

    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    for i in range(seqlen):
        data = train_raw if train_filters is None else filter(train_filters[i], train_raw)
        training_gen.append(HandwrittenNumberGenerator(
            mnist=data,
            ndigits=ndigits,
            digit_shape=digit_shape,
            supervised=supervised,
            dataset_size=size,
            batch_size=batch_size,
            leading_zeros=leading_zeros,
            infinite_gen=infinite_gen,
            random_state=random_state if infinite_gen else random_state.randint(2**32-1)
        ))

    for i in range(seqlen):
        data = test_raw if test_filters is None else filter(test_filters[i], train_raw)
        test_gen.append(HandwrittenNumberGenerator(
            mnist=data,
            ndigits=ndigits,
            digit_shape=digit_shape,
            supervised=supervised,
            dataset_size=size,
            batch_size=batch_size,
            leading_zeros=leading_zeros,
            infinite_gen=infinite_gen,
            random_state=random_state if infinite_gen else random_state.randint(2**32-1)
        ))

    for i in range(seqlen):
        data = train_raw if train_filters is None else filter(train_filters[i], train_raw)
        validation_gen.append(HandwrittenNumberGenerator(
            mnist=data,
            ndigits=ndigits,
            digit_shape=digit_shape,
            supervised=supervised,
            # validation set size as a fraction of training set size
            dataset_size=int(0.1 * size),
            batch_size=batch_size,
            leading_zeros=leading_zeros,
            infinite_gen=infinite_gen,
            random_state=random_state if infinite_gen else random_state.randint(2**32-1)
        ))

    training_set = SequenceGenerator(training_gen)
    test_set = SequenceGenerator(test_gen)
    validation_set = SequenceGenerator(validation_gen)

    return training_set, validation_set, test_set
