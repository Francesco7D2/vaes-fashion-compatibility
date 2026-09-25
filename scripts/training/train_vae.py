"""Train the original reconstruction objective using the shared model."""

import argparse
import json
import math
import os
import pickle
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from src.data_processing.image_processing import parse_tfrecord_fn
from src.models.vae256 import Conv_VAE
from src.utils.device_utilities import configure_tensorflow
from src.utils.setup_utilities import ensure_parent, file_digest, load_config
from src.utils.train_utilities import generate_and_save_images, train_step


def train_outfit_embeddings(config_path=None, device=None, show_progress=None):
    config = load_config(config_path)
    data, training = config["data"], config["training"]
    for key in ("epochs", "batch_size", "checkpoint_every"):
        value = training.get(key, 20 if key == "checkpoint_every" else 0)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if (data["image_height"], data["image_width"]) != (128, 96):
        raise ValueError("The original model requires 128 x 96 product images")
    device_report = configure_tensorflow(device or training.get("device", "auto"))
    print(f"TensorFlow device: {json.dumps(device_report)}", flush=True)
    tf.keras.utils.set_random_seed(training.get("seed", 42))
    dataset = tf.data.TFRecordDataset(
        training["train_dataset_path"], num_parallel_reads=tf.data.AUTOTUNE
    ).map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        lambda image: tf.ensure_shape(image, (128, 960, 3)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(training["batch_size"]).prefetch(tf.data.AUTOTUNE)
    outfit_count = len(pd.read_csv(data["outfits_w_products_path"], usecols=[0]))
    batches_per_epoch = math.ceil(outfit_count / training["batch_size"])
    if show_progress is None:
        show_progress = training.get("progress_bar", True)
    samples = list(
        tf.data.TFRecordDataset(training["train_sample_path"]).map(parse_tfrecord_fn)
    )
    samples = tf.stack(samples) if samples else None
    model = Conv_VAE()
    model(tf.zeros((1, 128, 960, 3)))
    optimizer = tf.keras.optimizers.Adam(training.get("learning_rate", 1e-4))
    history = []
    epoch_seconds = []
    print(
        f"Training {outfit_count} outfits in {batches_per_epoch} batches per epoch "
        f"for {training['epochs']} epochs.",
        flush=True,
    )
    if show_progress:
        print(
            "The first batch compiles the TensorFlow graph; wait for several batches "
            "before trusting its ETA.",
            flush=True,
        )
    for epoch in range(1, training["epochs"] + 1):
        with tqdm(
            dataset,
            total=batches_per_epoch,
            desc=f"Epoch {epoch}/{training['epochs']}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        ) as progress:
            start = time.monotonic()
            metric = tf.keras.metrics.Mean()
            for batch in progress:
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
            f"seconds={epoch_seconds[-1]:.1f}",
            flush=True,
        )
        if (
            epoch % training.get("checkpoint_every", 20) == 0
            or epoch == training["epochs"]
        ):
            model.save_weights(ensure_parent(training["model_weights_path"]))
            encoding_batches = tqdm(
                dataset,
                total=batches_per_epoch,
                desc=f"Checkpoint {epoch}: encoding",
                unit="batch",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            try:
                embeddings = tf.concat(
                    [model.encode(batch)[0] for batch in encoding_batches], axis=0
                ).numpy()
            finally:
                encoding_batches.close()
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
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"))
    parser.add_argument(
        "--progress", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args()
    train_outfit_embeddings(args.config, args.device, args.progress)
