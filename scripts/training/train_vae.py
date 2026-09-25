"""Train the original reconstruction objective using the shared model."""

import argparse
import json
import pickle
import time

import pandas as pd
import tensorflow as tf

from src.data_processing.image_processing import parse_tfrecord_fn
from src.models.vae256 import Conv_VAE
from src.utils.setup_utilities import ensure_parent, file_digest, load_config
from src.utils.train_utilities import generate_and_save_images, train_step


def train_outfit_embeddings(config_path=None):
    config = load_config(config_path)
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
    samples = list(
        tf.data.TFRecordDataset(training["train_sample_path"]).map(parse_tfrecord_fn)
    )
    samples = tf.stack(samples) if samples else None
    model = Conv_VAE()
    model(tf.zeros((1, 128, 960, 3)))
    optimizer = tf.keras.optimizers.Adam(training.get("learning_rate", 1e-4))
    history = []
    epoch_seconds = []
    for epoch in range(1, training["epochs"] + 1):
        start = time.monotonic()
        metric = tf.keras.metrics.Mean()
        for batch in dataset:
            metric.update_state(
                train_step(model, batch, optimizer),
                sample_weight=tf.cast(tf.shape(batch)[0], tf.float32),
            )
        if not int(metric.count.numpy()):
            raise ValueError("Training dataset is empty")
        history.append(float(metric.result().numpy()))
        epoch_seconds.append(time.monotonic() - start)
        print(
            f"Epoch {epoch}: reconstruction_loss={history[-1]:.6f}; "
            f"seconds={epoch_seconds[-1]:.1f}"
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
                        "epoch_seconds": epoch_seconds,
                        "embedding_type": "posterior_mean",
                    },
                    stream,
                )
            pd.DataFrame(
                {
                    "epoch": range(1, len(history) + 1),
                    "reconstruction_loss": history,
                    "epoch_seconds": epoch_seconds,
                }
            ).to_csv(ensure_parent(training["metrics_path"]), index=False)
            manifest = {
                "objective": "mean_pixel_binary_cross_entropy_without_kl",
                "embedding_type": "posterior_mean",
                "epoch": epoch,
                "sha256": {
                    key: file_digest(path)
                    for key, path in {
                        "weights": training["model_weights_path"],
                        "training_records": training["train_dataset_path"],
                        "training_outfits": data["outfits_w_products_path"],
                    }.items()
                },
            }
            manifest_path = training.get(
                "model_manifest_path", training["model_weights_path"] + ".json"
            )
            with open(ensure_parent(manifest_path), "w") as stream:
                json.dump(manifest, stream, indent=2)
            if samples is not None:
                generate_and_save_images(
                    model, epoch, samples, training["epoch_images_path"]
                )
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    train_outfit_embeddings(parser.parse_args().config)
