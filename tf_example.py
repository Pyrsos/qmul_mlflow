import numpy as np
import os
import tensorflow as tf
import mlflow
import mlflow.tensorflow

remote_server_uri = "http://0.0.0.0:5000"
mlflow.set_tracking_uri(remote_server_uri)

mlflow.set_experiment('MNIST_TF')

mlflow.tensorflow.autolog()


def conv_layer(input_layer, n_filters, kernel_size, activation, dropout_rate):
    out = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, activation=activation)(input_layer)
    out = tf.keras.layers.MaxPooling2D()(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.BatchNormalization()(out)

    return out


def hidden_layer(input_layer, layer_size, activation, dropout_rate):
    out = tf.keras.layers.Dense(layer_size, activation=activation)(input_layer)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.BatchNormalization()(out)

    return out


# Define some hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 512
N_FILTERS = [128, 64, 32]
ACTIVATIONS = ['relu', 'relu', 'relu']
KERNEL_SIZES = [3, 3, 3]
DROPOUT_RATES = [0.5, 0.4, 0.3]
HIDDEN_LAYER_SIZES = [400, 200]
HIDDEN_LAYER_ACTIVATIONS = ['relu', 'relu']
HIDDEN_LAYER_DROPOUT_RATES = [0.3, 0.2]
EARLY_STOP_PATIENCE = 3
REDUCE_LR_PATIENCE = 3
MONITOR_METRIC = "val_loss"

# Log some hyperparametes
mlflow.log_param("batch_size", BATCH_SIZE)
mlflow.log_param("monitor_metric", MONITOR_METRIC)
mlflow.log_param("reduce_lr_patience", REDUCE_LR_PATIENCE)
mlflow.log_param("n_filters", N_FILTERS)
mlflow.log_param("conv_activations", ACTIVATIONS)
mlflow.log_param("kernel_sizes", KERNEL_SIZES)
mlflow.log_param("conv_dropout_rates", DROPOUT_RATES)
mlflow.log_param("hidden_layer_sizes", HIDDEN_LAYER_SIZES)
mlflow.log_param("hidden_layer_activations", HIDDEN_LAYER_ACTIVATIONS)
mlflow.log_param("hidden_layer_dropout_rates", HIDDEN_LAYER_DROPOUT_RATES)


def cnn_model(input_shape):
    input_layer = tf.keras.layers.Input(shape=input_shape)
    out = tf.keras.layers.BatchNormalization()(input_layer)
    for filter_size, kernel_size, activation, dropout_rate in zip(N_FILTERS, KERNEL_SIZES, ACTIVATIONS, DROPOUT_RATES):
        out = conv_layer(out, filter_size, kernel_size, activation, dropout_rate)

    out = tf.keras.layers.Flatten()(out)
    for hidden_layer_size, activation, dropout_rate in zip(HIDDEN_LAYER_SIZES, HIDDEN_LAYER_ACTIVATIONS, HIDDEN_LAYER_DROPOUT_RATES):
        out = hidden_layer(out, hidden_layer_size, activation, dropout_rate)

    out = tf.keras.layers.Dense(10, activation='softmax')(out)

    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model = tf.keras.models.Model(inputs=input_layer, outputs=out)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return model


def training():
    # Get data
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
    train_labels = tf.keras.utils.to_categorical(train_labels)
    test_labels = tf.keras.utils.to_categorical(test_labels)

    # Reshape data for convolutional network.
    if N_FILTERS:
        train_images = np.reshape(train_images, (train_images.shape[0], 28, 28, 1))
        test_images = np.reshape(test_images, (test_images.shape[0], 28, 28, 1))

    # Define some callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor=MONITOR_METRIC, patience=EARLY_STOP_PATIENCE, verbose=1, mode='min', restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=MONITOR_METRIC, factor=0.1, patience=REDUCE_LR_PATIENCE, verbose=1, epsilon=1e-4, mode='min'),
    ]

    model = cnn_model(train_images.shape[1:])
    model.fit(train_images, train_labels, validation_data=(test_images, test_labels),
              epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

    best_epoch_loss, best_epoch_acc, best_epoch_precision, best_epoch_recall = model.evaluate(test_images, test_labels)

    mlflow.log_metric("best_epoch_loss", best_epoch_loss)
    mlflow.log_metric("best_epoch_acc", best_epoch_acc)
    mlflow.log_metric("best_epoch_precision", best_epoch_precision)
    mlflow.log_metric("best_epoch_recall", best_epoch_recall)

if __name__ == "__main__":
    training()
