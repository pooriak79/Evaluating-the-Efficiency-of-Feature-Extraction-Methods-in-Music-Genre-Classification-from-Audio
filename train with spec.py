

# =================================================
# 1) تنظیمات اولیه برای استفاده بهینه از CPU / GPU
# =================================================
import tensorflow as tf
import os

num_threads =50
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

# بررسی و فعال‌سازی GPU (درصورت وجود)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # فعال‌سازی تدریجی حافظه GPU (مانع از اشغال کل VRAM در ابتدا)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("[INFO] GPU memory growth enabled.")
        print("[INFO] Running on GPU:", physical_devices[0])
    except Exception as e:
        print("[WARNING] Could not set memory growth:", e)
else:
    print(f"[INFO] No accessible GPU found; using CPU with {num_threads} threads.")

# =================================================
# 2) ایمپورت سایر کتابخانه‌های مورد نیاز
# =================================================
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# =================================================
# 3) تابع پیشرفته بارگذاری داده و تبدیل به اسپکتروگرام
# =================================================
def load_and_preprocess_data(
    data_dir,
    classes,
    chunk_duration=4.0,
    overlap_duration=2.0,
    sr=None,
    n_fft=2048,
    hop_length=512,
    window='hann',
    use_db_scale=True,
    target_shape=(150, 150),
    padding_mode='constant',
    normalize=False,
    trim_silence=False,
    top_db=20,
    verbose=True
):
    """
    این تابع فایل‌های صوتی را از فولدر هر کلاس (در مسیر data_dir) می‌خواند، آنها را
    به قطعات کوچک‌تر تقسیم کرده و به اسپکتروگرام تبدیل می‌کند.

    پارامترهای کلیدی:
      - chunk_duration / overlap_duration: تعیین اندازه و هم‌پوشانی قطعه صوتی (ثانیه)
      - sr=None => حفظ نرخ نمونه‌برداری فایل اصلی
      - n_fft, hop_length, window: پارامترهای STFT
      - use_db_scale=True => تبدیل دامنه به dB
      - target_shape=(150, 150): اندازه نهایی اسپکتروگرام
      - normalize=True => نرمال‌سازی 0 تا 1
      - trim_silence=True => حذف سکوت ابتدا/انتهای فایل
      - top_db=20 => تنظیم میزان حذف سکوت
    """
    data = []
    labels = []

    if verbose:
        print("=== Audio Preprocessing Configuration ===")
        print(f"data_dir       : {data_dir}")
        print(f"classes        : {classes}")
        print(f"chunk_duration : {chunk_duration}s, overlap: {overlap_duration}s")
        print(f"sr             : {sr} (None means use file sr)")
        print(f"n_fft          : {n_fft}")
        print(f"hop_length     : {hop_length}")
        print(f"window         : {window}")
        print(f"use_db_scale   : {use_db_scale}")
        print(f"target_shape   : {target_shape}")
        print(f"padding_mode   : {padding_mode}")
        print(f"normalize      : {normalize}")
        print(f"trim_silence   : {trim_silence} (top_db={top_db})")
        print("=========================================\n")

    for class_index, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if verbose:
            print(f"[INFO] Processing class: {class_name} (folder: {class_dir})")

        if not os.path.isdir(class_dir):
            if verbose:
                print(f"[WARNING] Skipping: {class_dir} is not a directory.")
            continue

        for filename in os.listdir(class_dir):
            if not filename.lower().endswith('.wav'):
                continue  # فقط فایل‌های WAV

            file_path = os.path.join(class_dir, filename)
            if verbose:
                print(f"    Loading file: {filename}", end=' ... ')

            # تلاش برای بارگذاری فایل WAV
            try:
                audio_data, sample_rate = librosa.load(file_path, sr=sr)
            except Exception as e:
                if verbose:
                    print(f"\n[ERROR] Could not load {filename}: {e}")
                continue

            if verbose:
                print(f"Done. sr={sample_rate}, length={len(audio_data)} samples.")

            # حذف سکوت ابتدا و انتها در صورت نیاز
            if trim_silence:
                trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=top_db)
                if len(trimmed_audio) > 0:
                    audio_data = trimmed_audio

            # محاسبه نمونه‌های قطعه / اورلپ
            chunk_samples = int(chunk_duration * sample_rate)
            overlap_samples = int(overlap_duration * sample_rate)

            # رد کردن فایل‌های کوتاه‌تر از اندازه‌ی قطعه
            if len(audio_data) < chunk_samples:
                if verbose:
                    print(f"    [WARNING] Skipping {filename}: length < {chunk_samples} samples.")
                continue

            # تعداد قطعات
            num_chunks = int(np.ceil((len(audio_data) - chunk_samples) 
                                     / (chunk_samples - overlap_samples))) + 1
            if verbose:
                print(f"    -> Splitting into {num_chunks} chunks...")

            for i_chunk in range(num_chunks):
                start = i_chunk * (chunk_samples - overlap_samples)
                end = start + chunk_samples
                chunk = audio_data[start:end]

                # اگر این chunk کوتاه است (مثلاً آخرین تکه)، صفرپر می‌کنیم
                short_fall = chunk_samples - len(chunk)
                if short_fall > 0:
                    chunk = np.pad(chunk, (0, short_fall), mode=padding_mode)

                # محاسبه STFT و اسپکتروگرام
                stft_result = librosa.stft(
                    chunk,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window
                )
                spectrogram = np.abs(stft_result)

                # در صورت نیاز تبدیل به dB
                if use_db_scale:
                    spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

                # افزودن بعد کانال (H,W) -> (H,W,1)
                spectrogram = np.expand_dims(spectrogram, axis=-1)

                # تغییر اندازه به ابعاد موردنظر
                spectrogram = resize(spectrogram, (*target_shape,))

                # نرمال‌سازی در صورت فعال بودن
                if normalize:
                    min_val = spectrogram.min()
                    max_val = spectrogram.max()
                    if (max_val - min_val) != 0:
                        spectrogram = (spectrogram - min_val) / (max_val - min_val)
                    else:
                        spectrogram = np.zeros_like(spectrogram)

                data.append(spectrogram)
                labels.append(class_index)

        if verbose:
            print(f"[INFO] Finished class: {class_name}\n")

    # تبدیل به آرایه‌های Numpy
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    if verbose:
        print(f"Final Data shape: {data.shape}")
        print(f"Final Labels shape: {labels.shape}")

    return data, labels


# =================================================
# 4) تعیین مسیر داده و نام کلاس‌ها
# =================================================
# لطفاً مسیر لوکال دیتاست خود را قرار دهید:
data_dir = r"C:\Users\KHOOBTEK\Desktop\Music Genre Classification System\Data\genres_original"  # یا مسیر مناسب دیگر
classes = [
    'blues', 'classical','country','disco','hiphop',
    'jazz','metal','pop','reggae','rock'
]

# =================================================
# 5) بارگذاری و پیش‌پردازش داده
# =================================================
data, labels = load_and_preprocess_data(
    data_dir=data_dir,
    classes=classes,
    chunk_duration=4.0,
    overlap_duration=2.0,
    sr=None,            # نرخ نمونه‌برداری فایل اصلی حفظ شود
    n_fft=2048,
    hop_length=512,
    window='hann',
    use_db_scale=True,
    target_shape=(150, 150),
    padding_mode='constant',
    normalize=False,
    trim_silence=False, # درصورت تمایل، True شود
    top_db=20,
    verbose=True
)

# تبدیل برچسب‌ها به One-Hot
num_classes = len(classes)
labels = to_categorical(labels, num_classes=num_classes)

# =================================================
# 6) جدا کردن داده به آموزش و تست
# =================================================
X_train, X_test, y_train, y_test = train_test_split(
    data, labels,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("[INFO] Data split:")
print("  Train shape:", X_train.shape, y_train.shape)
print("  Test  shape:", X_test.shape,  y_test.shape)


# =================================================
# 7) ساخت مدل CNN پیشرفته
# =================================================
model = Sequential()

# بلوک اول
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',
                 input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# بلوک دوم
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# بلوک سوم
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.3))

# بلوک چهارم
model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# بلوک پنجم
model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

model.add(Dropout(0.3))

# لایه‌های پایانی
model.add(Flatten())
model.add(Dense(units=1200, activation='relu'))
model.add(Dropout(0.45))
model.add(Dense(units=num_classes, activation='softmax'))

model.summary()

# =================================================
# 8) کامپایل مدل
# =================================================
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =================================================
# 9) آموزش مدل
# =================================================
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# ذخیره مدل
model.save("Trained_model.h5")
print("[INFO] Model saved to Trained_model.h5")

# =================================================
# 10) ارزیابی مدل
# =================================================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("\n=== Final Evaluation ===")
print(f"Train accuracy: {train_acc:.4f} | Train loss: {train_loss:.4f}")
print(f"Test  accuracy: {test_acc:.4f}  | Test  loss: {test_loss:.4f}")

# =================================================
# 11) رسم نمودار Loss و Accuracy
# =================================================
epochs = range(1, len(history.history['loss']) + 1)

plt.figure(figsize=(10,4))
plt.plot(epochs, history.history['loss'], 'r-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'b-', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(epochs, history.history['accuracy'], 'r-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'b-', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.show()

# =================================================
# 12) گزارش طبقه‌بندی و ماتریس درهم‌ریختگی
# =================================================
y_pred = model.predict(X_test)
predicted_categories = np.argmax(y_pred, axis=1)
true_categories = np.argmax(y_test, axis=1)

print("\n=== Classification Report ===")
print(classification_report(true_categories, predicted_categories, target_names=classes))

cm = confusion_matrix(true_categories, predicted_categories)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Class')
plt.ylabel('Actual Class')
plt.title('Confusion Matrix')
plt.show()
