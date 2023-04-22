import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import time

# 損失関数
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Generatorモデル
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(
            1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(
            2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(
            2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model


# Discriminatorモデル
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2),
                      padding='same', input_shape=[28, 28, 1]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model


# 損失関数定義(識別器)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# 損失関数定義(生成器)
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# 生成された画像の表示
def display_generated_images(epoch, generator):
    seed = tf.random.normal([16, 100])
    predictions = generator(seed, training=False)

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    st.pyplot(fig)
    plt.close(fig)


def display_loss_graph(generator_losses, discriminator_losses):
    plt.plot(generator_losses, label="Generator Loss")
    plt.plot(discriminator_losses, label="Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    st.pyplot(plt.gcf())


# 学習プロセス
def train(epochs, batch_size):

    # データセットの準備
    (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype("float32")

    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5
    buffer_size = train_images.shape[0]
    train_dataset = tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(buffer_size).batch(batch_size)

    # モデルの定義
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # オプティマイザと損失関数の定義
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # エポックごとの損失を格納するリスト
    generator_losses = []
    discriminator_losses = []

    # プログレスバーの設定
    progress_bar = st.progress(0)

    # 学習プロセスの実行
    for epoch in range(epochs):
        st.write(f"### Epoch {epoch + 1} 学習開始")
        start = time.time()
        gen_loss_sum = 0
        disc_loss_sum = 0
        num_batches = 0

        for image_batch in train_dataset:
            noise = tf.random.normal([batch_size, 100])
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(image_batch, training=True)
                fake_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(fake_output)
                disc_loss = discriminator_loss(real_output, fake_output)

                gen_loss_sum += gen_loss.numpy()
                disc_loss_sum += disc_loss.numpy()
                num_batches += 1

            gradients_of_generator = gen_tape.gradient(
                gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(
                disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(
                zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.trainable_variables))

        # 学習後の画像表示
        display_generated_images(epoch, generator)

        # 損失の平均値を計算し、リストに追加
        generator_losses.append(gen_loss_sum / num_batches)
        discriminator_losses.append(disc_loss_sum / num_batches)

        # エポックごとの時間計測
        st.write(f"Epoch {epoch + 1} 完了, 時間: {time.time() - start:.2f}秒")

        # プログレスバーの更新
        progress_bar.progress((epoch + 1) / epochs)

    # 学習が終わった後の損失関数のグラフ表示
    display_loss_graph(generator_losses, discriminator_losses)


# Streamlitアプリ
def main():
    st.title("DCGAN体験アプリ")
    st.write("※実行環境がCPUのため、1epoch辺り大体10分近くかかります。")

    st.sidebar.header("学習前のMNIST画像")
    (X_train, _), (_, _) = mnist.load_data()
    num_images_to_show = 8
    X_train = np.expand_dims(X_train[:num_images_to_show], axis=-1)

    # 画像を横に連結します
    concatenated_images = np.concatenate(X_train, axis=1)

    # 連結された画像を表示します
    st.sidebar.image(concatenated_images, width=28 *
                     num_images_to_show, caption="MNIST画像の例")

    st.sidebar.title("設定")
    epochs = st.sidebar.number_input(
        "エポック数", min_value=1, max_value=100, value=50, step=1)
    batch_size = st.sidebar.number_input(
        "バッチサイズ", min_value=32, max_value=256, value=64, step=32)

    st.sidebar.subheader("学習開始")
    if st.sidebar.button("学習を開始"):
        train(epochs, batch_size)


if __name__ == "__main__":
    main()
