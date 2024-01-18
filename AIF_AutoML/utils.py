import os
import cv2
import glob
import inspect
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds  # https://www.tensorflow.org/datasets/catalog/overview?hl=ko
from tqdm import tqdm
from PIL import Image
import scipy.signal as signal
from scipy.interpolate import RegularGridInterpolator

tf.get_logger().setLevel('ERROR')


def load_data(name='mnist', split='train', verbose=True):
    try:
        ds, info = tfds.load(name=name, split=split, with_info=True)
        label_names = info.features['label'].names

        X, Y = [], []
        for xy in tqdm(tfds.as_numpy(ds)):
            X.append(xy['image'])
            Y.append(tf.keras.utils.to_categorical(xy['label'], num_classes=len(label_names), dtype='float32'))

    except Exception as e:
        print(f'Error: {e}')

    if verbose:
        try:
            print(name, f'X length (shape): {len(X)} ({X[0].shape}),'f'Y length (shape): {len(Y)} ({Y[0].shape})')
        except Exception as e:
            print(name, f'X length: {len(X)},'f'Y length: {len(Y)}')

    return X, Y, label_names, info


def load_image_data(dir_base_path='datasets/mnist', split='train', shuffle=True, random_seed=42, verbose=True):
    try:
        X, Y = [], []
        dir_split_path = os.path.join(dir_base_path, split)
        indices_label_names = sorted([x for x in os.listdir(dir_split_path) if '.' not in x])
        if len(indices_label_names) > 0:
            indice = [int(x.split('_')[0]) for x in indices_label_names]
            label_names = [x.split('_')[1] for x in indices_label_names]

            for index_label_name, index in zip(indices_label_names, indice):
                dir_image_path = f'{dir_split_path}/{index_label_name}'
                path_image_list = sorted(glob.glob(f'{dir_image_path}/*'))
                for path_image in path_image_list:
                    X.append(np.array(Image.open(path_image)))
                    Y.append(tf.keras.utils.to_categorical(index, num_classes=len(label_names), dtype='float32'))
                print('Loaded %d images from %s' % (len(path_image_list), dir_image_path))
        else:
            filename_list = sorted(os.listdir(dir_split_path))
            for filename in tqdm(filename_list):
                X.append(np.array(Image.open(os.path.join(dir_split_path, filename))))
                Y.append(float(filename.split('_')[0]))
            label_names = ''

        info = {'base_path': dir_base_path, 'split': split, 'label_names': label_names}

        if shuffle:
            dataset = list(zip(X, Y))
            np.random.seed(random_seed)
            np.random.shuffle(dataset)
            X, Y = zip(*dataset)

        if verbose:
            if label_names == '':
                print(dir_split_path, f'X count (shape): {len(X)} ({X[0].shape}), Y count (shape): {len(Y)} ({Y[0]})')
            else:
                print(dir_split_path, f'X count (shape): {len(X)} ({X[0].shape}), Y count (shape): {len(Y)} ({Y[0].shape})')

        return X, Y, label_names, info

    except Exception as e:
        print(f'Error: {e}')


def load_signal_data(dir_base_path='datasets/sigpeak', split='train', shuffle=True, random_seed=42, verbose=True):
    try:
        T, X, Y = [], [], []
        dir_split_path = os.path.join(dir_base_path, split)
        indices_label_names = sorted([x for x in os.listdir(dir_split_path) if '.' not in x])
        if len(indices_label_names) > 0:
            indice = [int(x.split('_')[0]) for x in indices_label_names]
            label_names = [x.split('_')[1] for x in indices_label_names]

            for index_label_name, index in zip(indices_label_names, indice):
                dir_signal_path = f'{dir_split_path}/{index_label_name}'
                path_signal_list = sorted(glob.glob(f'{dir_signal_path}/*'))
                for path_signal in path_signal_list:
                    T.append(np.loadtxt(path_signal, delimiter=',', skiprows=1)[:, 0])
                    X.append(np.loadtxt(path_signal, delimiter=',', skiprows=1)[:, 1])
                    Y.append(tf.keras.utils.to_categorical(index, num_classes=len(label_names), dtype='float32'))
                print('Loaded %d signals from %s' % (len(path_signal_list), dir_signal_path))
        else:
            filename_list = sorted(os.listdir(dir_split_path))
            for filename in tqdm(filename_list):
                T.append(np.loadtxt(os.path.join(dir_split_path, filename), delimiter=',', skiprows=1)[:, 0])
                X.append(np.loadtxt(os.path.join(dir_split_path, filename), delimiter=',', skiprows=1)[:, 1])
                Y.append(float(filename.split('_')[0]))
            label_names = ''

        info = {'base_path': dir_base_path, 'split': split, 'label_names': label_names}

        if shuffle:
            dataset = list(zip(T, X, Y))
            np.random.seed(random_seed)
            np.random.shuffle(dataset)
            T, X, Y = zip(*dataset)

        if verbose:
            if label_names == '':
                print(dir_split_path, f'T count (shape): {len(T)} ({T[0].shape}), X count (shape): {len(X)} ({X[0].shape}), Y count (shape): {len(Y)} ({Y[0]})')
            else:
                print(dir_split_path, f'T count (shape): {len(T)} ({T[0].shape}), X count (shape): {len(X)} ({X[0].shape}), Y count (shape): {len(Y)} ({Y[0].shape})')

        return T, X, Y, label_names, info

    except Exception as e:
        print(f'Error: {e}')


# plot image channelwisely
def plot_images(images, labels, label_names, i_list=[]):
    try:
        images = np.array(images, dtype=np.uint8)
        labels = np.array(labels)
    except:
        pass
    i_list = range(len(images)) if i_list == [] else i_list
    grayscale = False if images[0].shape[-1] == 3 else True

    if grayscale:
        fig = plt.figure(figsize=(3 * len(i_list), 3))
    else:
        fig = plt.figure(figsize=(3 * 4, 3 * len(i_list)))

    ax_list = []
    for i_ax, i_image in enumerate(i_list):
        x = images[i_image]
        y = labels[i_image].astype(np.float16)

        try:
            y = f'{label_names[np.argmax(y)]} {y}'
        except:
            pass

        if grayscale:
            ax = fig.add_subplot(1, len(i_list), i_ax + 1)
            ax.imshow(x, cmap='gray')
            ax.set_title(f"Image #{i_image}\nLabel: {y}\nShape: {x.shape}")
            ax.set_axis_off()
        else:
            for i_ax_column in range(4):
                if i_ax_column == 0:
                    ax = fig.add_subplot(len(i_list), 4, i_ax * 4 + 1)
                    ax.imshow(x)
                    ax.set_title(f"Image #{i_image}\nLabel: {y}\nShape: {x.shape}", loc='left')
                    ax.set_axis_off()
                else:
                    i_channel = i_ax_column - 1
                    x_channel = x.copy()
                    x_channel[:, :, [x for x in range(3) if x != i_channel]] = 0
                    ax = fig.add_subplot(len(i_list), 4, i_ax * 4 + 1 + i_ax_column)
                    ax.imshow(x_channel)
                    ax.set_title(f"{['Red', 'Green', 'Blue'][i_channel]} Channel")
                    ax.set_axis_off()
        ax_list.append(ax)

    fig.tight_layout()
    fig.show()


# plot signals
def plot_signals(axis, signals, labels, label_names, i_list=[], axis_name='Time [s]'):
    try:
        axis = np.array(axis)
        signals = np.array(signals)
        labels = np.array(labels)
    except:
        pass
    i_list = range(len(signals)) if i_list == [] else i_list

    fig = plt.figure(figsize=(6, 3 * len(i_list)))

    ax_list = []
    for i_ax, i_signal in enumerate(i_list):
        a = axis[i_signal]
        x = signals[i_signal]
        y = labels[i_signal].astype(np.float16)

        try:
            y = f'{label_names[np.argmax(y)]} {y}'
        except:
            pass

        ax = fig.add_subplot(len(i_list), 1, i_ax + 1)
        ax.plot(a, x * 0, 'r')
        ax.plot(a, x, linewidth=5)
        ax.set_xlim(a[0], a[-1])
        ax.set_xlabel(axis_name)
        ax.set_ylabel('Value')
        ax.set_title(f"Image #{i_signal}\nLabel: {y}\nShape: {x.shape}", loc='left')
        ax.grid()

        ax_list.append(ax)

    fig.tight_layout()
    fig.show()


# prompt: python code to convert rgb image to grayscale image based on opencv
def rgb_to_grayscale_image(image):
    try:
        if image.shape[-1] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            print('Error: The input image is not a RGB image')
            return image

    except Exception as e:
        print(f'Error: {e}')
        return image


# prompt: python code to resize and fill image based on opencv
def resize_and_fill_image(image, target_size, fill_color=(0, 0, 0)):
    try:
        if image.shape[-1] == 1:
            new_image = np.full((target_size[0], target_size[1]), fill_color[0], dtype=np.uint8)
        else:
            new_image = np.full((target_size[0], target_size[1], 3), fill_color, dtype=np.uint8)

        scale = min(target_size[0] / image.shape[0], target_size[1] / image.shape[1])
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        center = (target_size[0] // 2, target_size[1] // 2)
        position = (center[0] - resized_image.shape[0] // 2, center[1] - resized_image.shape[1] // 2)
        new_image[position[0]:position[0] + resized_image.shape[0], position[1]:position[1] + resized_image.shape[1]] = resized_image
        return new_image

    except Exception as e:
        print(f'Error: {e}')
        return image


# prompt: python code to resize and crop image based on opencv
def resize_and_crop_image(image, target_size):
    try:
        scale = max(target_size[0] / image.shape[0], target_size[1] / image.shape[1])
        resized_image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        start_x = (resized_image.shape[1] - target_size[1]) // 2
        start_y = (resized_image.shape[0] - target_size[0]) // 2
        end_x = start_x + target_size[1]
        end_y = start_y + target_size[0]
        cropped_image = resized_image[start_y:end_y, start_x:end_x]
        return cropped_image

    except Exception as e:
        print(f'Error: {e}')
        return image


def flip_image(image, mode='lr'):  # lr: left-right, ud: up-down
    try:
        if mode == 'lr':
            flipped_image = np.fliplr(image)
        elif mode == 'ud':
            flipped_image = np.flipud(image)
        else:
            print('Error: The input mode is not valid (lr for left-right, ud for up-down)')
            return image
        return flipped_image

    except Exception as e:
        print(f'Error: {e}')
        return image


# prompt: python code to rotate image based on opencv
def rotate_image(image, angle_list=[], fill_color=(255, 255, 255)):
    try:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        angle = np.random.randint(0, 360) if angle_list == [] else -angle_list[np.random.randint(0, len(angle_list))]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]), borderValue=fill_color)
        return rotated_image

    except Exception as e:
        print(f'Error: {e}')
        return image


def mixup_images(images, labels, alpha=0.5):
    try:
        images, labels = np.array(images), np.array(labels)
    except:
        assert False, 'The input images should be same shape'

    indices_random = np.random.permutation(images.shape[0])
    images_shuffled, labels_shuffled = images[indices_random], labels[indices_random]

    ratio = np.random.beta(alpha, alpha, images.shape[0]) if alpha > 0 else 1.0

    ratio = ratio.reshape(-1, 1, 1, 1)
    images_mixed = ratio * images + (1 - ratio) * images_shuffled

    ratio = ratio.reshape(-1, 1)
    labels_mixed = ratio * labels + (1 - ratio) * labels_shuffled

    return images_mixed, labels_mixed


def cutmix_images(images, labels, beta=0.5):
    try:
        images, labels = np.array(images), np.array(labels)
    except:
        assert False, 'The input images should be same shape'

    indices_random = np.random.permutation(len(images))
    target_a, target_b = labels, labels[indices_random]

    h, w = images[0].shape[:2]
    ratio_cut = np.sqrt(1. - np.random.beta(beta, beta))
    w_cut, h_cut = int(w * ratio_cut), int(h * ratio_cut)
    x_cut, y_cut = np.random.randint(w), np.random.randint(h)
    bbx1, bby1 = np.clip(x_cut - w_cut // 2, 0, w), np.clip(y_cut - h_cut // 2, 0, h)
    bbx2, bby2 = np.clip(x_cut + w_cut // 2, 0, w), np.clip(y_cut + h_cut // 2, 0, h)

    images_mixed = images.copy()
    images_mixed[:, bbx1:bbx2, bby1:bby2, :] = images[indices_random, bbx1:bbx2, bby1:bby2, :]

    ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (h * w))
    labels_mixed = target_a * ratio + target_b * (1. - ratio)

    return images_mixed, labels_mixed


def absolute_signal(signal_data):
    absolute_signal = np.abs(signal_data)
    return absolute_signal


def min_max_scaling_signal(signal_data):
    min_val = min(signal_data)
    max_val = max(signal_data)
    scaled_signal = (signal_data - min_val) / (max_val - min_val)
    return scaled_signal


def standardize_signal(signal_data):
    mean = np.mean(signal_data)
    std = np.std(signal_data)
    standardized_signal = (signal_data - mean) / std
    return standardized_signal


def moving_average_signal(signal_data, window_size):
    window = np.ones(window_size) / window_size
    smoothed_signal = np.convolve(signal_data, window, mode='same')
    return smoothed_signal


def linear_detrend_signal(signal_data):
    detrended_signal = signal_data - np.linspace(signal_data[0], signal_data[-1], len(signal_data))
    return detrended_signal


def remove_outlier(signal_data, threshold=3.5):
    median = np.median(signal_data)
    diff = np.abs(signal_data - median)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation if med_abs_deviation != 0 else 0
    modified_z_score[modified_z_score == 0] = 1e-18
    indices = np.where(modified_z_score > threshold)
    signal_data[indices] = median
    return signal_data


def lowpass_filter(signal_data, time_delta, cutoff_freq, order=5):
    sample_rate = 1 / time_delta
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='low')
    filtered_signal = signal.lfilter(b, a, signal_data)
    return filtered_signal


def highpass_filter(signal_data, time_delta, cutoff_freq, order=5):
    sample_rate = 1 / time_delta
    nyquist_freq = 0.5 * sample_rate
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = signal.butter(order, normalized_cutoff_freq, btype='high')
    filtered_signal = signal.lfilter(b, a, signal_data)
    return filtered_signal


def compute_fft(signal_data, time_delta):
    n = len(signal_data)
    f = np.fft.fftfreq(n, d=time_delta)[:n // 2]
    mag = np.fft.fft(signal_data)
    mag = 2.0 / n * np.abs(mag[0:n // 2])
    return f, mag


def compute_psd(signal_data, time_delta):
    sample_rate = 1 / time_delta
    f, psd = signal.welch(signal_data, sample_rate)
    return f, psd


def compute_spectrogram(signal_data, time_delta, time_max, nperseg, s_width=100, s_height=100):
    signal_data = signal_data.squeeze()  # Remove the channel dimension if it exists (num_channels, t_length) -> (t_length)

    sampling_rate = 1 / time_delta  # Sampling frequency

    Sf, St, Sxx = signal.spectrogram(signal_data, nperseg=nperseg)

    t_axis = np.linspace(0, time_max, Sxx.shape[1])
    f_axis = np.linspace(0, sampling_rate / 2, Sxx.shape[0])  # sr / 2 is the Nyquist frequency

    grid_t_axis = np.linspace(0, time_max, s_width)
    grid_f_axis = np.linspace(0, sampling_rate / 2, s_height)  # sr / 2 is the Nyquist frequency

    interp_func = RegularGridInterpolator((f_axis, t_axis), Sxx)

    grid_f, grid_t = np.meshgrid(grid_f_axis, grid_t_axis, indexing='ij')  # Create the 2D grid
    points = np.stack((grid_f.ravel(), grid_t.ravel()), axis=-1)  # Create the points to interpolate by ravel() that returns a flattened array
    S_resized = interp_func(points).reshape(s_height, s_width)  # Interpolate the points
    S_resized = np.flipud(S_resized)  # Flip the image to have the origin at the top

    return grid_f, grid_t, S_resized


def write_func_to_py_file(func, file_path):
    with open(file_path, 'w') as file:
        file.write(inspect.getsource(func))


def create_submission_zipfile(submission_name='mnist_hogeony_pp1', model_best=None, preprocessing=None):
    write_func_to_py_file(preprocessing, f'{submission_name}.py')
    print(f'Created {submission_name}.py')

    model_best.save(f'{submission_name}.h5')
    print(f'Created {submission_name}.h5')

    with zipfile.ZipFile(f'{submission_name}.zip', 'w') as zipf:
        zipf.write(f'{submission_name}.py')
        zipf.write(f'{submission_name}.h5')
    print(f'Created {submission_name}.zip')
