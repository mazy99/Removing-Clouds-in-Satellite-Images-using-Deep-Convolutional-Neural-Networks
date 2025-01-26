import tensorflow as tf
import numpy as np
import rasterio
import os
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras.optimizers import Adam

# Генератор: U-Net
def build_generator(output_channels):
    return pix2pix.unet_generator(output_channels=output_channels, norm_type='batchnorm')

# Дискриминатор: PatchGAN
def build_discriminator():
    return pix2pix.discriminator(norm_type='batchnorm', target=False)

# Потери генератора
def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))  # L1 Loss
    total_loss = gan_loss + (lambda_l1 * l1_loss)
    return total_loss

# Потери дискриминатора
def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    return real_loss + generated_loss

# Загрузка изображения из TIFF
def load_tiff_image(file_path, target_size=(256, 256)):
    with rasterio.open(file_path) as src:
        image = src.read().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    image = tf.image.resize(image, target_size)  # Изменяем размер
    image = tf.cast(image, tf.float32) / 65535.0  # Нормализация (TIFF может быть 16-битным)
    return image

# Загрузка маски из NPY
def load_npy_mask(file_path, target_size=(256, 256)):
    mask = np.load(file_path)  # Загружаем маску
    mask = tf.image.resize(mask[..., np.newaxis], target_size, method='nearest')  # Изменяем размер
    mask = tf.cast(mask, tf.float32)
    return mask

# Создание датасета
def load_data(image_dir, mask_dir, target_size=(256, 256)):
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.npy')])

    images = [load_tiff_image(img, target_size) for img in image_files]
    masks = [load_npy_mask(mask, target_size) for mask in mask_files]

    return tf.data.Dataset.from_tensor_slices((images, masks)).batch(4)

# Визуализация результата
def visualize_predictions(generator, test_dataset):
    for cloudy_image, clear_image in test_dataset.take(1):
        prediction = generator(cloudy_image, training=True)
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        ax[0].imshow(cloudy_image[0][:, :, 0], cmap='gray')  # Первый канал
        ax[0].set_title("Облачное изображение")
        ax[1].imshow(clear_image[0][:, :, 0], cmap='gray')  # Первый канал
        ax[1].set_title("Целевая маска")
        ax[2].imshow(prediction[0][:, :, 0], cmap='gray')  # Первый канал
        ax[2].set_title("Сгенерированное изображение")
        plt.show()

# Основная функция обучения
def train(image_dir, mask_dir, output_channels=15, epochs=50, lambda_l1=100):
    generator = build_generator(output_channels)
    discriminator = build_discriminator()

    generator_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(2e-4, beta_1=0.5)

    dataset = load_data(image_dir, mask_dir)

    for epoch in range(epochs):
        print(f"Эпоха {epoch + 1}/{epochs}")
        for cloudy_image, clear_image in dataset:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Генерация изображения
                generated_image = generator(cloudy_image, training=True)

                # Оценка дискриминатором
                real_output = discriminator([cloudy_image, clear_image], training=True)
                fake_output = discriminator([cloudy_image, generated_image], training=True)

                # Потери
                gen_loss = generator_loss(fake_output, generated_image, clear_image, lambda_l1)
                disc_loss = discriminator_loss(real_output, fake_output)

            # Обновление весов
            gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
            disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

        # Визуализация каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            print(f"Результаты после эпохи {epoch + 1}:")
            visualize_predictions(generator, dataset)

    print("Обучение завершено.")
    return generator

# Запуск
if __name__ == "__main__":
    image_dir = "data/tiff_images"  # Папка с TIFF изображениями
    mask_dir = "data/npy_masks"    # Папка с NPY масками

    # Убедитесь, что данные загружены в указанные директории
    generator_model = train(image_dir, mask_dir)
    print("Генератор успешно обучен.")
