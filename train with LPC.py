#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Music Genre Classification with LPC + Delta + Delta2 + CMVN features
Keeping the exact same CNN architecture (no changes in layers/filters),
and using Data Augmentation with phase_vocoder for time-stretch.

Author: YourName
Date  : 2024/xx/xx
"""

import os
import random
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- spafe برای استخراج LPC ---
try:
    from spafe.features.lpc import lpc
except ImportError:
    print("[ERROR] spafe library not installed. Please install via: pip install spafe")
    exit(1)

from tensorflow.keras.layers import (Conv2D, MaxPool2D, Flatten, Dense, 
                                     Dropout, Activation)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


# ========================================
# 1. تنظیم منابع (CPU/GPU) با تعداد Threads
# ========================================
num_threads = 50
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[INFO] GPU memory growth enabled.")
        print("[INFO] Running on GPU:", physical_devices[0])
    except Exception as e:
        print("[WARNING] Could not set GPU memory growth:", e)
else:
    print(f"[INFO] No accessible GPU found; using CPU with {num_threads} threads.")


# ================================
# 2. پارامترهای مربوط به دیتاست
# ================================
DATA_DIR = "Data/genres_original"  # فولدر دیتاست
CLASSES = ['blues', 'classical','country','disco','hiphop',
           'jazz','metal','pop','reggae','rock']

CHUNK_DURATION   = 4.0  # طول هر چانک (ثانیه)
OVERLAP_DURATION = 2.0  # همپوشانی (ثانیه)
TARGET_SHAPE     = (150, 150)  # اندازهٔ نهایی ویژگی (ارتفاع=150، عرض=150)
LPC_ORDER        = 12   # مرتبهٔ LPC
# (تعداد ضرایب LPC خروجی، طبق پیاده‌سازی spafe، می‌شود order+1 در هر فریم)


# ========================================
# 3. توابع کمکی برای پردازش و Augmentation
# ========================================
def cmvn(feature_array):
    """
    اعمال Cepstral Mean and Variance Normalization (CMVN)
    روی ماتریس ورودی: شکل (num_features, time_frames).
    """
    mean = feature_array.mean(axis=1, keepdims=True)
    std  = feature_array.std(axis=1, keepdims=True) + 1e-9
    return (feature_array - mean) / std

def phase_vocoder_time_stretch(y, rate=1.0):
    """
    کشش زمانی سیگنال موج با استفاده از فاز-ووکدر librosa:
    1) محاسبه STFT
    2) اعمال phase_vocoder
    3) بازگشت به حوزهٔ زمانی با iSTFT
    """
    if rate == 1.0:
        return y  # بدون تغییر

    # تبدیل سیگنال خام به فرکانس-زمان
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    # کشش زمانی
    D_stretched = librosa.phase_vocoder(D, rate=rate, hop_length=512)
    # بازگشت به حوزهٔ زمان
    y_stretched = librosa.istft(D_stretched, hop_length=512)
    return y_stretched

def augment_audio(y, sr, p=0.5):
    """
    با احتمال p، اعمالی مانند نویز، Pitch Shift و Time Stretch
    روی سیگنال y به‌صورت تصادفی اعمال می‌شود.
    اینجا از فاز-ووکدر librosa برای کشش زمانی استفاده می‌کنیم.
    """
    import random

    # اطمینان از اینکه y نوع float دارد
    if not np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.float32)

    if random.random() > p:
        return y  # بدون تغییر

    # 1. افزودن نویز تصادفی با احتمال 0.3
    if random.random() < 0.3:
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        y = y + noise_amp * np.random.normal(size=len(y))

    # 2. تغییر گام (Pitch Shift) با احتمال 0.3
    if random.random() < 0.3:
        n_steps = np.random.uniform(-2, 2)  # ممکن است بین -2 تا +2 نیم‌پرده
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

    # 3. تغییر سرعت پخش (Time Stretch) با احتمال 0.3
    if random.random() < 0.3:
        rate = np.random.uniform(0.8, 1.2)
        y = phase_vocoder_time_stretch(y, rate=rate)

    return y


# ======================================================
# 4. تابع استخراج ویژگی LPC + Delta + Delta2 + CMVN
# ======================================================
import numpy as np
import librosa
from spafe.features.lpc import lpc
from scipy.linalg import LinAlgError

def extract_lpc_features(
    chunk,
    sr,
    order=12,
    apply_cmvn_=True,
    resize_shape=(150, 150),
    normalize=False
):
    try:
        # گام 1) استخراج LPC با spafe
        lpc_all = lpc(sig=chunk, fs=sr, order=order)  # ممکن است tuple برگردد
        if isinstance(lpc_all, tuple):
            lpc_feat = lpc_all[0]  # اگر تاپل بود، بخش اول همان ضرایب LPC
        else:
            lpc_feat = lpc_all
    except LinAlgError:
        # اگر ماتریس تکین شود، یک ماتریس پیش‌فرض صفر برگردانیم:
        lpc_feat = np.zeros((1, order + 1), dtype=np.float32)

    # اگر آرایه صفر باشد یا نرمال نیست:
    if lpc_feat.size == 0:
        lpc_feat = np.zeros((1, order + 1), dtype=np.float32)

    # ادامه مراحل دلتا و دلتا2 و غیره
    lpc_delta  = librosa.feature.delta(lpc_feat, order=1, axis=1)
    lpc_delta2 = librosa.feature.delta(lpc_feat, order=2, axis=1)

    lpc_cat = np.concatenate([lpc_feat, lpc_delta, lpc_delta2], axis=0)

    if apply_cmvn_:
        lpc_cat = cmvn(lpc_cat)

    lpc_cat = np.expand_dims(lpc_cat, axis=-1)
    lpc_resized = resize(lpc_cat, (*resize_shape, 1))

    if normalize:
        _min, _max = lpc_resized.min(), lpc_resized.max()
        if (_max - _min) > 1e-9:
            lpc_resized = (lpc_resized - _min) / (_max - _min)
        else:
            lpc_resized = np.zeros_like(lpc_resized)

    return lpc_resized.astype(np.float32)



# ==============================================================================
# 5. تابع کلی برای بارگذاری فایل‌ها، برش به چانک، اعمال Augmentation، استخراج LPC
# ==============================================================================
def load_and_preprocess_data(
    data_dir,
    classes,
    chunk_duration=4.0,
    overlap_duration=2.0,
    sr=None,
    augmentation=True,
    trim_silence=False,
    top_db=20,
    verbose=True
):
    data = []
    labels = []

    if verbose:
        print("[INFO] Loading dataset from:", data_dir)
        print(f"[INFO] Classes: {classes}")
        print(f"[INFO] chunk_duration={chunk_duration}s, overlap={overlap_duration}s")
        print(f"[INFO] augmentation={augmentation}, trim_silence={trim_silence}")

    for class_idx, class_name in enumerate(classes):
        class_folder = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_folder):
            print(f"[WARNING] Skip {class_folder}: not a directory.")
            continue

        if verbose:
            print(f"\n[INFO] Processing class: {class_name}")

        for file_name in os.listdir(class_folder):
            if not file_name.lower().endswith(".wav"):
                continue

            file_path = os.path.join(class_folder, file_name)
            if verbose:
                print(f"   -> Loading {file_name} ... ", end='')

            try:
                audio_data, sr_loaded = librosa.load(file_path, sr=sr)
            except Exception as e:
                print(f"\n[ERROR] Could not load file: {file_name}, error: {e}")
                continue

            # اگر صحیح لود شد
            if verbose:
                print("OK.", end=" ")

            # تریم سکوت ابتدا/انتها (اختیاری)
            if trim_silence:
                trimmed_data, _ = librosa.effects.trim(audio_data, top_db=top_db)
                if len(trimmed_data) > 0:
                    audio_data = trimmed_data

            chunk_samples    = int(chunk_duration   * sr_loaded)
            overlap_samples  = int(overlap_duration * sr_loaded)

            # اگر فایل کوتاه‌تر از یک چانک باشد
            if len(audio_data) < chunk_samples:
                if verbose:
                    print(f"[WARNING] {file_name} is shorter than {chunk_samples} samples. Skipped.")
                continue

            # محاسبه تعداد چانک‌ها
            num_chunks = int(np.ceil(
                (len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples)
            )) + 1

            if verbose:
                print(f" => Splitting into {num_chunks} chunks.")

            for i_chunk in range(num_chunks):
                start = i_chunk * (chunk_samples - overlap_samples)
                end   = start + chunk_samples
                chunk = audio_data[start:end]

                # پدینگ در صورت کوتاه بودن چانک آخر
                if len(chunk) < chunk_samples:
                    pad_size = chunk_samples - len(chunk)
                    chunk = np.pad(chunk, (0, pad_size), mode='constant')

                # اعمال Data Augmentation (در صورت فعال بودن)
                if augmentation:
                    chunk = augment_audio(chunk, sr_loaded, p=0.4)

                # استخراج ویژگی با LPC
                feature_data = extract_lpc_features(
                    chunk, sr=sr_loaded,
                    order=LPC_ORDER,
                    apply_cmvn_=True,
                    resize_shape=TARGET_SHAPE,
                    normalize=False
                )

                data.append(feature_data)
                labels.append(class_idx)

    data   = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if verbose:
        print("\n[INFO] Finished processing.")
        print(f"       Data shape : {data.shape}")
        print(f"       Labels shape : {labels.shape}")
        if data.size > 0:
            print(f"       Data range  : {data.min()} to {data.max()}")

    return data, labels


# =================================================
# 6. ساخت مدل با همان معماری دلخواه شما (بدون تغییر)
# =================================================
def build_model(input_shape, num_classes):

    from tensorflow.keras.models import Sequential

    model = Sequential()
    
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.3))

    model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=2, strides=2))

    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(units=1200, activation='relu'))
    model.add(Dropout(0.45))

    # لایه خروجی
    model.add(Dense(units=num_classes, activation='softmax'))

    return model


# =========================
# 7. قسمت اصلی اجرا (main)
# =========================
def main():
    # --- الف: آماده‌سازی داده ---
    data, labels = load_and_preprocess_data(
        data_dir=DATA_DIR,
        classes=CLASSES,
        chunk_duration=CHUNK_DURATION,
        overlap_duration=OVERLAP_DURATION,
        sr=None,            # None => استفاده از نرخ نمونه برداری فایل
        augmentation=True,  # اعمال Data Augmentation
        trim_silence=False, # حذف سکوت ابتدا/انتها (درصورت نیاز True کنید)
        top_db=20,
        verbose=True
    )

    # تبدیل لیبل‌ها به one-hot
    labels = to_categorical(labels, num_classes=len(CLASSES))

    # تقسیم آموزش/آزمون
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels,
        test_size=0.2,
        random_state=42
    )

    print("\n[INFO] Split Data:")
    print(f"   X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"   X_test shape : {X_test.shape},  y_test shape : {y_test.shape}")

    # --- ب: ساخت مدل با معماری بدون تغییر ---
    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASSES))
    model.summary()

    # کامپایل
    optimizer = Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # --- ج: آموزش ---
    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # ذخیره مدل
    model.save("Trained_model_LPC.h5")

    # ذخیره تاریخچه آموزش (اختیاری)
    with open("training_history_lpc.json", "w") as f:
        json.dump(history.history, f)

    # --- ارزیابی ---
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc   = model.evaluate(X_test, y_test, verbose=0)

    print(f"\n[RESULT] Training Accuracy: {train_acc:.4f}")
    print(f"[RESULT] Testing Accuracy : {test_acc:.4f}")

    # رسم نمودار Loss و Accuracy
    plot_learning_curves(history)

    # گزارش و ماتریس درهم‌ریختگی
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASSES))

    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.show()


def plot_learning_curves(history):
    """
    رسم نمودارهای Loss و Accuracy
    """
    epochs = range(1, len(history.history['loss']) + 1)

    # نمودار Loss
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.history['loss'], label='Train Loss', color='red')
    plt.plot(epochs, history.history['val_loss'], label='Val Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.show()

    # نمودار Accuracy
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history.history['accuracy'], label='Train Accuracy', color='red')
    plt.plot(epochs, history.history['val_accuracy'], label='Val Accuracy', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()


# ====================================
# اجرای اصلی در صورت اجرای مستقیم فایل
# ====================================
if __name__ == "__main__":
    main()
