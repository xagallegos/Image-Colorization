import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def normalize_image(image):
    return tf.cast(image, tf.float32) / 255.0


def create_dataset(pth, batch=64):
    gray = tf.keras.utils.image_dataset_from_directory(pth, image_size=(128,128), batch_size=batch, labels=None, shuffle=False, color_mode='grayscale')
    color = tf.keras.utils.image_dataset_from_directory(pth, image_size=(128,128), batch_size=batch, labels=None, shuffle=False, color_mode='rgb')
    
    gray = gray.map(normalize_image)
    color = color.map(normalize_image)

    dataset = tf.data.Dataset.zip((gray, color))

    return dataset


def create_model():
    latent_dim = 256

    # encoder
    encoder_input = layers.Input(shape=(128,128,1), name='encoder_input')
    x = layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(encoder_input)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(x)

    shape = K.int_shape(x)

    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name='latent_vector')(x)
    encoder = Model(encoder_input, latent, name='encoder')

    # decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='decoder_input')
    x = layers.Dense(shape[1]*shape[2]*shape[3])(decoder_input)
    x = layers.Reshape((shape[1], shape[2], shape[3]))(x)
    x = layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    output = layers.Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

    decoder = Model(decoder_input, output, name='decoder')

    # autoencoder
    autoencoder = Model(encoder_input, decoder(encoder(encoder_input)))
    
    return autoencoder

def main():
    train_path = "../../../../Downloads/archive/val_256/train_samples"
    val_path = "../../../../Downloads/archive/val_256/valid_samples"

    train_dataset = create_dataset(train_path)
    val_dataset = create_dataset(val_path)

    autoencoder = create_model()

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_path = "models/training_2/cp.ckpt"

    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)
    
    autoencoder.fit(train_dataset,
                validation_data = val_dataset,
                epochs = 30,
                batch_size = 64,
                callbacks = [cp_callback])

if __name__ == '__main__':
    main()
    