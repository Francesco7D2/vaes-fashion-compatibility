"""Train the original reconstruction objective using the shared model."""

import math
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.data_processing.image_processing import parse_tfrecord_fn
from src.models.vae256 import Conv_VAE
from src.utils.setup_utilities import ensure_parent, load_config
from src.utils.train_utilities import generate_and_save_images, train_step


def train_outfit_embeddings(config_path=None, overrides=None):
    config = load_config(config_path, overrides)
    data, training = config["data"], config["training"]
    for key in ("epochs", "batch_size", "checkpoint_every"):
        value = training.get(key, 20 if key == "checkpoint_every" else 0)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (data["image_height"], data["image_width"]) != (128, 96):
        raise ValueError("The original model requires 128 x 96 product images")
    tf.keras.utils.set_random_seed(training.get("seed", 42))
    dataset = tf.data.TFRecordDataset(training["train_dataset_path"]).map(
        parse_tfrecord_fn
    )
    dataset = dataset.map(lambda image: tf.ensure_shape(image, (128, 960, 3)))
    dataset = dataset.batch(training["batch_size"]).prefetch(tf.data.AUTOTUNE)
    num_batches = math.ceil(
        sum(1 for _ in tf.data.TFRecordDataset(training["train_dataset_path"]))
        / training["batch_size"]
    )
    samples = list(
        tf.data.TFRecordDataset(training["train_sample_path"]).map(parse_tfrecord_fn)
    )
    samples = tf.stack(samples) if samples else None
    model = Conv_VAE()
    model(tf.zeros((1, 128, 960, 3)))
    optimizer = tf.keras.optimizers.Adam(training.get("learning_rate", 1e-4))
    history = []
    devices = [device.name for device in tf.config.list_physical_devices("GPU")]
    print(f"Training on {devices[0] if devices else 'CPU'}; {num_batches} batches per epoch")
    for epoch in range(1, training["epochs"] + 1):
        start = time.monotonic()
        metric = tf.keras.metrics.Mean()
        progress = tqdm(
            dataset,
            total=num_batches,
            desc=f"Epoch {epoch}/{training['epochs']}",
            unit="batch",
            leave=False,
        )
        for batch in progress:
            metric.update_state(
                train_step(model, batch, optimizer),
                sample_weight=tf.cast(tf.shape(batch)[0], tf.float32),
            )
            progress.set_postfix(loss=f"{metric.result().numpy():.5f}")
        progress.close()
        if not int(metric.count.numpy()):
            raise ValueError("Training dataset is empty")
        history.append(float(metric.result().numpy()))
        tqdm.write(
            f"Epoch {epoch}/{training['epochs']}: reconstruction_loss={history[-1]:.6f}; "
            f"seconds={time.monotonic() - start:.1f}"
        )
        if (
            epoch % training.get("checkpoint_every", 20) == 0
            or epoch == training["epochs"]
        ):
            model.save_weights(ensure_parent(training["model_weights_path"]))
            embeddings = tf.concat(
                [model.encode(batch)[0] for batch in dataset], axis=0
            ).numpy()
            with open(ensure_parent(training["model_data_path"]), "wb") as stream:
                pickle.dump(
                    {
                        "latent_space": embeddings,
                        "reconstruction_loss": history,
                        "embedding_type": "posterior_mean",
                    },
                    stream,
                )
            if samples is not None:
                generate_and_save_images(
                    model, epoch, samples, training["epoch_images_path"]
                )
    return model, history


if __name__ == "__main__":
    train_outfit_embeddings()
