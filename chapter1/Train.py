import os
import cv2
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === CONFIGURAZIONE ===
IMG_SIZE = 256
BASE_DIR = "/content/drive/MyDrive/branch/VOCdevkit/VOC2012"
IMAGES_DIR = os.path.join(BASE_DIR, "JPEGImages")
MASKS_DIR = os.path.join(BASE_DIR, "SegmentationClass")

print("🎯 U-NET - SEGMENTAZIONE RAMI")
print("=" * 60)

# === FUNZIONI DATASET (già testate) ===
def get_valid_image_mask_pairs():
    print("🔍 Ricerca coppie immagine-maschera valide...")
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

    print(f"   Trovate {len(image_files)} immagini e {len(mask_files)} maschere")

    valid_pairs = []
    for image_file in image_files:
        image_id = os.path.splitext(image_file)[0]
        mask_file = image_id + '.png'
        mask_path = os.path.join(MASKS_DIR, mask_file)

        if os.path.exists(mask_path):
            valid_pairs.append({
                'id': image_id,
                'image_path': os.path.join(IMAGES_DIR, image_file),
                'mask_path': mask_path
            })

    print(f"   ✅ Coppie valide trovate: {len(valid_pairs)}")
    return valid_pairs

def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Impossibile caricare l'immagine: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return img

def load_and_preprocess_mask_corrected(mask_path):
    mask = cv2.imread(mask_path)
    if mask is None:
        raise ValueError(f"Impossibile caricare la maschera: {mask_path}")

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_resized = cv2.resize(mask_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    binary_mask = np.any(mask_resized != [0, 0, 0], axis=-1).astype(np.float32)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    return binary_mask

def create_dataset(pairs, split_name):
    print(f"\n📦 Creazione dataset {split_name}...")

    X, y = [], []
    failed_pairs = []

    for pair in tqdm(pairs):
        try:
            img = load_and_preprocess_image(pair['image_path'])
            mask = load_and_preprocess_mask_corrected(pair['mask_path'])
            X.append(img)
            y.append(mask)
        except Exception as e:
            failed_pairs.append((pair['id'], str(e)))
            continue

    if failed_pairs:
        print(f"   ⚠️  {len(failed_pairs)} coppie fallite")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"   ✅ {split_name} completato: {len(X)} immagini")
    return X, y

# === ARCHITETTURA U-NET ===
def build_unet(input_shape=(256, 256, 3)):
    """Costruisce architettura U-Net per segmentazione binaria"""
    print("🧠 Costruzione U-Net...")

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (Contraction Path)
    # Blocco 1
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Blocco 2
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Blocco 3
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # Blocco 4 (Bottleneck)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)

    # Decoder (Expansion Path)
    # Blocco 6
    u6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)

    # Blocco 7
    u7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)

    # Blocco 8
    u8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)

    # Blocco 9
    u9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs, name='U-Net')

    print("✅ U-Net costruita con successo!")
    model.summary()
    return model

# === LOSS FUNCTIONS ===
def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Calcola il Dice Coefficient"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice Loss per dataset sbilanciati"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred, alpha=0.5):
    """Combina Binary Crossentropy e Dice Loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dice

# === DATA AUGMENTATION ===
def create_augmentation_layer():
    """Crea layer di data augmentation"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1)
    ])

# === TRAINING ===
def train_unet_model(X_train, y_train, X_val, y_val, epochs=50):
    """Addestra il modello U-Net"""
    print("\n🏋️‍♂️ INIZIO TRAINING U-NET...")

    # Build model
    model = build_unet()

    # Compila il modello con metriche multiple
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[
            'accuracy',
            'Precision',
            'Recall',
            tf.keras.metrics.MeanIoU(num_classes=2),
            dice_coefficient
        ]
    )

    print("📋 Configurazione training:")
    print("   - Optimizer: Adam (lr=1e-4)")
    print("   - Loss: Combined (BCE + Dice)")
    print("   - Metrics: Accuracy, Precision, Recall, IoU, Dice")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_unet_model.h5',
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Data augmentation
    augmentation_layer = create_augmentation_layer()

    # Training con augmentation
    print("🔄 Applicazione data augmentation...")
    X_train_aug = augmentation_layer(X_train).numpy()

    # Training
    print("🚀 Avvio training...")
    history = model.fit(
        X_train_aug, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,  # Batch size più piccolo per migliore stabilità
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Training completato!")
    return model, history

# === VALUTAZIONE ===
def evaluate_model(model, X_val, y_val):
    """Valuta le performance del modello"""
    print("\n📊 VALUTAZIONE MODELLO...")

    # Valutazione quantitativa
    results = model.evaluate(X_val, y_val, verbose=0)

    print("📈 Metriche finali:")
    metrics = ['Loss', 'Accuracy', 'Precision', 'Recall', 'IoU', 'Dice']
    for metric, value in zip(metrics, results):
        print(f"   {metric}: {value:.4f}")

    # Calcola F1-Score
    precision = results[2]
    recall = results[3]
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"   F1-Score: {f1:.4f}")

    return results

def visualize_results(model, X_val, y_val, num_samples=5):
    """Visualizza predizioni vs ground truth"""
    print(f"\n👀 VISUALIZZAZIONE RISULTATI ({num_samples} campioni)...")

    # Seleziona campioni casuali
    indices = random.sample(range(len(X_val)), min(num_samples, len(X_val)))
    X_samples = X_val[indices]
    y_true_samples = y_val[indices]

    # Predizioni
    y_pred_samples = model.predict(X_samples, verbose=0)

    # Visualizzazione
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))

    for i in range(num_samples):
        # Immagine originale
        axes[i, 0].imshow(X_samples[i])
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Ground Truth
        axes[i, 1].imshow(y_true_samples[i].squeeze(), cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predizione
        axes[i, 2].imshow(y_pred_samples[i].squeeze(), cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Overlay
        axes[i, 3].imshow(X_samples[i])
        axes[i, 3].imshow(y_pred_samples[i].squeeze(), cmap='Reds', alpha=0.6)
        axes[i, 3].set_title('Overlay Prediction')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.show()

def plot_training_history(history):
    """Visualizza la storia del training"""
    print("\n📈 ANDAMENTO TRAINING...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Dice Coefficient
    axes[0, 2].plot(history.history['dice_coefficient'], label='Training Dice')
    axes[0, 2].plot(history.history['val_dice_coefficient'], label='Validation Dice')
    axes[0, 2].set_title('Dice Coefficient')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # IoU
    axes[1, 2].plot(history.history['mean_io_u'], label='Training IoU')
    axes[1, 2].plot(history.history['val_mean_io_u'], label='Validation IoU')
    axes[1, 2].set_title('Mean IoU')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

# === SCRIPT PRINCIPALE ===
def main():
    """Script principale - End to End U-Net Training"""
    print("🚀 U-NET - TRAINING COMPLETO")
    print("=" * 60)

    # 1. Caricamento dati
    print("\n1️⃣  FASE 1: CARICAMENTO DATI")
    valid_pairs = get_valid_image_mask_pairs()

    if len(valid_pairs) == 0:
        print("❌ Nessuna coppia valida trovata!")
        return

    # Split dataset
    train_pairs, val_pairs = train_test_split(
        valid_pairs, test_size=0.2, random_state=42, shuffle=True
    )

    print(f"🎯 Split dataset:")
    print(f"   Train: {len(train_pairs)} immagini")
    print(f"   Validation: {len(val_pairs)} immagini")

    # Creazione dataset
    X_train, y_train = create_dataset(train_pairs, "train")
    X_val, y_val = create_dataset(val_pairs, "validation")

    # 2. Training
    print("\n2️⃣  FASE 2: TRAINING U-NET")
    model, history = train_unet_model(X_train, y_train, X_val, y_val, epochs=50)

    # 3. Valutazione
    print("\n3️⃣  FASE 3: VALUTAZIONE")
    results = evaluate_model(model, X_val, y_val)

    # 4. Visualizzazioni
    print("\n4️⃣  FASE 4: VISUALIZZAZIONI")
    plot_training_history(history)
    visualize_results(model, X_val, y_val, num_samples=5)

    # 5. Riepilogo finale
    print("\n" + "=" * 60)
    print("🎉 U-NET TRAINING COMPLETATO!")
    print("📊 RIEPILOGO FINALE:")
    print(f"   • Modello salvato: best_unet_model.h5")
    print(f"   • Accuracy: {results[1]:.4f}")
    print(f"   • Precision: {results[2]:.4f}")
    print(f"   • Recall: {results[3]:.4f}")
    print(f"   • IoU: {results[4]:.4f}")
    print(f"   • Dice: {results[5]:.4f}")
    print(f"   • Pronto per segmentazione rami!")

    return model, history, X_train, y_train, X_val, y_val

# === ESECUZIONE ===
if __name__ == "__main__":
    # Verifica GPU
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ GPU disponibile: {len(tf.config.experimental.list_physical_devices('GPU')) > 0}")

    # Esegui training completo
    model, history, X_train, y_train, X_val, y_val = main()
