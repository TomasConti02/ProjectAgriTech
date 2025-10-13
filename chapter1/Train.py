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

# === FUNZIONI DATASET MIGLIORATE ===
def get_valid_image_mask_pairs():
    print("🔍 Ricerca coppie immagine-maschera valide...")
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')]
    mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.png')]

    print(f"   Trovate {len(image_files)} immagini e {len(mask_files)} maschere")

    valid_pairs = []
    for image_file in tqdm(image_files, desc="Analisi coppie"):
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
    """Versione migliorata per gestire meglio le maschere VOC"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Carica in scala di grigi
    if mask is None:
        raise ValueError(f"Impossibile caricare la maschera: {mask_path}")

    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # Per VOC, i rami sono tipicamente in classi diverse dallo sfondo (0)
    # Considera tutto ciò che non è sfondo come ramo
    binary_mask = (mask_resized > 0).astype(np.float32)
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    
    return binary_mask

def create_dataset(pairs, split_name):
    print(f"\n📦 Creazione dataset {split_name}...")

    X, y = [], []
    failed_pairs = []

    for pair in tqdm(pairs, desc=f"Processing {split_name}"):
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
        if len(failed_pairs) > 5:
            print(f"   Prime 5 errori: {failed_pairs[:5]}")

    if len(X) == 0:
        raise ValueError(f"Nessun dato valido per {split_name}!")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"   ✅ {split_name} completato: {len(X)} immagini")
    return X, y

# === ARCHITETTURA U-NET MIGLIORATA ===
def build_unet(input_shape=(256, 256, 3)):
    """Costruisce architettura U-Net con batch normalization"""
    print("🧠 Costruzione U-Net migliorata...")

    inputs = tf.keras.Input(shape=input_shape)

    # Encoder (Contraction Path) con BatchNorm
    # Blocco 1
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    p1 = layers.Dropout(0.1)(p1)

    # Blocco 2
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    p2 = layers.Dropout(0.1)(p2)

    # Blocco 3
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    p3 = layers.Dropout(0.2)(p3)

    # Blocco 4
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    p4 = layers.Dropout(0.2)(p4)

    # Bottleneck
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Dropout(0.3)(c5)

    # Decoder (Expansion Path)
    # Blocco 6
    u6 = layers.Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    u6 = layers.Dropout(0.2)(u6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    # Blocco 7
    u7 = layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    u7 = layers.Dropout(0.2)(u7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)

    # Blocco 8
    u8 = layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    u8 = layers.Dropout(0.1)(u8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)

    # Blocco 9
    u9 = layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    u9 = layers.Dropout(0.1)(u9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)

    # Output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    model = Model(inputs, outputs, name='U-Net-Improved')

    print("✅ U-Net migliorata costruita con successo!")
    model.summary()
    return model

# === LOSS FUNCTIONS MIGLIORATE ===
def dice_coefficient(y_true, y_pred, smooth=1.0):
    """Calcola il Dice Coefficient"""
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice Loss per dataset sbilanciati"""
    return 1 - dice_coefficient(y_true, y_pred)

def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Focal Loss per gestire lo squilibrio delle classi"""
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    BCE_EXP = tf.exp(-BCE)
    focal_loss = alpha * tf.pow(1 - BCE_EXP, gamma) * BCE
    return tf.reduce_mean(focal_loss)

def combined_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """Combina Focal Loss e Dice Loss"""
    focal = focal_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * focal + beta * dice

# === DATA AUGMENTATION MIGLIORATA ===
def augment_data(images, masks):
    """Applica data augmentation a batch di immagini e maschere"""
    augmented_images = []
    augmented_masks = []
    
    for img, mask in zip(images, masks):
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        
        # Random vertical flip
        if random.random() > 0.5:
            img = np.flipud(img)
            mask = np.flipud(mask)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, k=angle//90)
            mask = np.rot90(mask, k=angle//90)
        
        augmented_images.append(img)
        augmented_masks.append(mask)
    
    return np.array(augmented_images), np.array(augmented_masks)

# === TRAINING MIGLIORATO ===
def train_unet_model(X_train, y_train, X_val, y_val, epochs=50):
    """Addestra il modello U-Net con training migliorato"""
    print("\n🏋️‍♂️ INIZIO TRAINING U-NET MIGLIORATO...")

    # Build model
    model = build_unet()

    # Compila il modello con metriche multiple
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.MeanIoU(num_classes=2, name='iou'),
            dice_coefficient
        ]
    )

    print("📋 Configurazione training migliorata:")
    print("   - Optimizer: Adam (lr=1e-4)")
    print("   - Loss: Combined (Focal + Dice)")
    print("   - Metrics: Accuracy, Precision, Recall, IoU, Dice")

    # Callbacks migliorati
    callbacks = [
        EarlyStopping(
            monitor='val_dice_coefficient',
            patience=20,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_dice_coefficient',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1,
            mode='max'
        ),
        ModelCheckpoint(
            'best_unet_model.keras',  # Usa il nuovo formato
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        )
    ]

    # Training con data augmentation on-the-fly
    print("🚀 Avvio training con augmentation...")
    
    # Crea dataset TensorFlow per training più efficiente
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    
    def augment_fn(image, mask):
        # Applica flip orizzontale
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)
        
        # Applica flip verticale
        if tf.random.uniform(()) > 0.5:
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)
        
        # Random brightness
        image = tf.image.random_brightness(image, 0.1)
        # Random contrast
        image = tf.image.random_contrast(image, 0.9, 1.1)
        
        return image, mask

    train_dataset = train_dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(8).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(8).prefetch(tf.data.AUTOTUNE)

    # Training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
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
    results = model.evaluate(X_val, y_val, verbose=0, batch_size=8)

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
    axes[1, 2].plot(history.history['iou'], label='Training IoU')
    axes[1, 2].plot(history.history['val_iou'], label='Validation IoU')
    axes[1, 2].set_title('Mean IoU')
    axes[1, 2].legend()
    axes[1, 2].grid(True)

    plt.tight_layout()
    plt.show()

# === SCRIPT PRINCIPALE MIGLIORATO ===
def main():
    """Script principale - End to End U-Net Training"""
    print("🚀 U-NET - TRAINING COMPLETO MIGLIORATO")
    print("=" * 60)

    try:
        # 1. Caricamento dati
        print("\n1️⃣  FASE 1: CARICAMENTO DATI")
        valid_pairs = get_valid_image_mask_pairs()

        if len(valid_pairs) == 0:
            print("❌ Nessuna coppia valida trovata!")
            return None

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
        print("\n2️⃣  FASE 2: TRAINING U-NET MIGLIORATO")
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
        print(f"   • Modello salvato: best_unet_model.keras")
        print(f"   • Accuracy: {results[1]:.4f}")
        print(f"   • Precision: {results[2]:.4f}")
        print(f"   • Recall: {results[3]:.4f}")
        print(f"   • IoU: {results[4]:.4f}")
        print(f"   • Dice: {results[5]:.4f}")
        print(f"   • Pronto per segmentazione rami!")

        return model, history, X_train, y_train, X_val, y_val

    except Exception as e:
        print(f"❌ Errore durante l'esecuzione: {e}")
        import traceback
        traceback.print_exc()
        return None

# === ESECUZIONE ===
if __name__ == "__main__":
    # Verifica GPU
    print(f"✅ TensorFlow version: {tf.__version__}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"✅ GPU disponibile: {len(gpus) > 0}")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth abilitato per GPU")
        except RuntimeError as e:
            print(f"⚠️  Errore configurazione GPU: {e}")

    # Esegui training completo
    try:
        results = main()
        if results is not None:
            model, history, X_train, y_train, X_val, y_val = results
            
            # Esempio inference
            print("\n💡 ESEMPIO INFERENCE:")
            print("""
            # Per usare il modello su nuove immagini:
            new_image = load_and_preprocess_image("path/to/image.jpg")
            prediction = model.predict(np.expand_dims(new_image, axis=0))

            # Per salvare la maschera:
            mask = (prediction[0].squeeze() > 0.5).astype(np.uint8) * 255
            cv2.imwrite("maschera_output.png", mask)
            """)
        else:
            print("❌ Training non completato a causa di errori")
            
    except KeyboardInterrupt:
        print("\n⏹️  Training interrotto dall'utente")
    except Exception as e:
        print(f"❌ Errore non gestito: {e}")
        import traceback
        traceback.print_exc()
