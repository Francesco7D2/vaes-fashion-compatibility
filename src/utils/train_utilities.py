"""Reconstruction-only training; the original experiment has no KL term."""

from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf


@tf.function
def compute_loss(model, x):
    """Mean pixel binary cross entropy (not a negative ELBO)."""
    x = tf.cast(x, tf.float32)
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=model(x), labels=x)
    )


def train_step(model, x, optimizer):
    """Create optimizer state outside the graph, including for repeated runs."""
    if not model.built:
        model(x)
    if not optimizer.built:
        optimizer.build(model.trainable_variables)
    return _train_step(model, x, optimizer)


@tf.function
def _train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def generate_and_save_images(model, epoch, test_sample, path):
    """Save deterministic reconstructions of any nonempty sample batch."""
    mean, _ = model.encode(test_sample)
    predictions = model.decode(mean, apply_sigmoid=True)
    count = len(test_sample)
    if not count:
        return
    Path(path).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(count * 2, 1, figsize=(10, count * 2), squeeze=False)
    for index, (original, reconstructed) in enumerate(zip(test_sample, predictions)):
        for axis, image in zip(
            axes[2 * index : 2 * index + 2, 0], (original, reconstructed)
        ):
            axis.imshow(image)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(Path(path) / f"image_at_epoch_{epoch:04d}.png")
    plt.close(fig)
