"""
Autoencoder Model Builder Module
Handles model architecture creation and training
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim, encoding_dims=[128, 64, 32]):
    """
    Build autoencoder architecture

    Args:
        input_dim: number of input features
        encoding_dims: list of hidden layer sizes for encoder

    Returns:
        Compiled Keras autoencoder model
    """
    print("\n" + "="*60)
    print("BUILDING AUTOENCODER MODEL")
    print("="*60)

    # Encoder
    input_layer = layers.Input(shape=(input_dim,))

    encoded = input_layer
    for i, dim in enumerate(encoding_dims):
        encoded = layers.Dense(dim, activation='relu', name=f'encoder_{i+1}')(encoded)
        encoded = layers.Dropout(0.2)(encoded)

    # Bottleneck
    bottleneck = encoded

    # Decoder (mirror of encoder)
    decoded = bottleneck
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        decoded = layers.Dense(dim, activation='relu', name=f'decoder_{i+1}')(decoded)
        decoded = layers.Dropout(0.2)(decoded)

    # Output layer
    output_layer = layers.Dense(input_dim, activation='linear', name='output')(decoded)

    # Build model
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer, name='autoencoder')

    # Compile
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    print(autoencoder.summary())

    return autoencoder


def train_autoencoder(model, X_train, X_val, epochs=100, batch_size=256):
    """
    Train autoencoder model

    Args:
        model: Keras autoencoder model
        X_train: Training features (scaled)
        X_val: Validation features (scaled)
        epochs: Number of training epochs
        batch_size: Batch size for training

    Returns:
        Training history object
    """
    print("\n" + "="*60)
    print("TRAINING AUTOENCODER")
    print("="*60)

    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )

    # Train (autoencoder reconstructs input, so X=y)
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history
