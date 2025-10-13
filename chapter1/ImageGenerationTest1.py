import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === PATH ===
MODEL_PATH = "/content/drive/MyDrive/branch/models/unet_branch_segmentation_final_20251011_183040.keras"
TUE_IMMAGINI_DIR = "/content/drive/MyDrive/branch/tue_immagini"
def segmentazione_ibrida_bilanciata(image_path):
    """
    Approccio ibrido bilanciato - parametri ottimizzati
    """
    # Carica immagine
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # === FASE 1: PREDIZIONE MODELLO ===
    img_processed = cv2.resize(img_rgb, (256, 256))
    img_processed = img_processed.astype(np.float32) / 255.0

    prediction = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)[0].squeeze()
    model_mask = (prediction > 0.15).astype(np.uint8)  # Soglia leggermente più alta
    model_mask = cv2.resize(model_mask, (original_w, original_h))

    # === FASE 2: ELABORAZIONE TRADIZIONALE BILANCIATA ===

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # PARAMETRI BILANCIATI PER RAMI:

    # 1. Segmentazione colore - PIÙ PERMISSIVA MA SELETTIVA
    # Range per marroni/rami (HSV) - allargato strategicamente
    lower_brown1 = np.array([0, 30, 10])     # Più permissivo su tonalità
    upper_brown1 = np.array([30, 220, 150])  # Più permissivo su valore

    lower_brown2 = np.array([150, 30, 10])   # Toni rossastri-marroni
    upper_brown2 = np.array([180, 200, 150])

    mask_hsv1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask_hsv2 = cv2.inRange(hsv, lower_brown2, upper_brown2)

    # 2. Segmentazione luminosità (LAB) - più permissiva
    L, A, B = cv2.split(lab)
    mask_lab_light = cv2.inRange(L, 15, 80)   # Range più ampio

    # 3. Segmentazione basata sul canale A/B LAB per colori naturali
    mask_lab_color = cv2.inRange(A, 110, 135)  # Colori naturali/vegetazione

    # 4. Edge detection bilanciata
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    # Edge detection con soglie medie
    edges = cv2.Canny(gray_eq, 40, 100)

    # 5. Analisi texture semplificata
    kernel = cv2.getGaborKernel((15, 15), 4.0, 0, 10.0, 1.5, 0, ktype=cv2.CV_32F)
    texture_filtered = cv2.filter2D(gray_eq, cv2.CV_8UC3, kernel)
    _, texture_mask = cv2.threshold(texture_filtered, 30, 255, cv2.THRESH_BINARY)

    # === FASE 3: COMBINAZIONE STRATEGICA ===

    # Combina maschere colore (OR tra diverse strategie)
    color_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    color_mask = cv2.bitwise_or(color_mask, mask_lab_light)
    color_mask = cv2.bitwise_or(color_mask, mask_lab_color)

    # Combina struttura (edge + texture)
    structure_mask = cv2.bitwise_or(edges, texture_mask)

    # Combinazione FINALE: colore E struttura
    traditional_mask = cv2.bitwise_and(color_mask, structure_mask)

    # Post-processing tradizionale
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    traditional_mask = cv2.morphologyEx(traditional_mask, cv2.MORPH_OPEN, kernel_clean)
    traditional_mask = cv2.morphologyEx(traditional_mask, cv2.MORPH_CLOSE, kernel_clean)

    # === FASE 4: COMBINAZIONE CON AI ===
    final_mask = cv2.bitwise_or(traditional_mask, model_mask * 255)

    # Pulizia finale
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)

    # Filtro area ragionevole
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 30 or area > (original_h * original_w * 0.4):
            final_mask[labels == i] = 0

    return final_mask, traditional_mask, model_mask * 255

def analisi_visuale_dettagliata(image_path):
    """
    Analisi visiva dettagliata per debugging
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Spazi colore per analisi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    H, S, V = cv2.split(hsv)

    final_mask, traditional_mask, model_mask = segmentazione_ibrida_bilanciata(image_path)

    if final_mask is None:
        return

    # Calcola statistiche
    area_percent = np.sum(final_mask > 0) / final_mask.size * 100
    area_traditional = np.sum(traditional_mask > 0) / traditional_mask.size * 100
    area_model = np.sum(model_mask > 0) / model_mask.size * 100

    print(f"📊 AREE SEGMENTATE:")
    print(f"   • Modello AI: {area_model:.2f}%")
    print(f"   • Tradizionale: {area_traditional:.2f}%")
    print(f"   • Ibrido Finale: {area_percent:.2f}%")

    # Visualizzazione COMPLETA per debugging
    plt.figure(figsize=(25, 8))

    # Riga 1: Immagini originali e spazi colore
    plt.subplot(2, 6, 1)
    plt.imshow(img_rgb)
    plt.title('Originale RGB')
    plt.axis('off')

    plt.subplot(2, 6, 2)
    plt.imshow(H, cmap='hsv')
    plt.title('Canale H (HSV)')
    plt.axis('off')

    plt.subplot(2, 6, 3)
    plt.imshow(S, cmap='gray')
    plt.title('Canale S (HSV)')
    plt.axis('off')

    plt.subplot(2, 6, 4)
    plt.imshow(V, cmap='gray')
    plt.title('Canale V (HSV)')
    plt.axis('off')

    plt.subplot(2, 6, 5)
    plt.imshow(L, cmap='gray')
    plt.title('Canale L (LAB)')
    plt.axis('off')

    plt.subplot(2, 6, 6)
    plt.imshow(A, cmap='gray')
    plt.title('Canale A (LAB)')
    plt.axis('off')

    # Riga 2: Risultati segmentazione
    plt.subplot(2, 4, 5)
    plt.imshow(model_mask, cmap='gray')
    plt.title(f'Modello AI\n{area_model:.1f}%')
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(traditional_mask, cmap='gray')
    plt.title(f'Tradizionale\n{area_traditional:.1f}%')
    plt.axis('off')

    plt.subplot(2, 4, 7)
    plt.imshow(final_mask, cmap='gray')
    plt.title(f'Ibrido Finale\n{area_percent:.1f}%')
    plt.axis('off')

    plt.subplot(2, 4, 8)
    plt.imshow(img_rgb)
    plt.imshow(final_mask, cmap='Reds', alpha=0.6)
    plt.title('Overlay Finale')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Consiglio parametri
    if area_traditional < 5:
        print("💡 SUGGERIMENTO: I rami potrebbero avere colori diversi dal marrone")
        print("   Prova ad analizzare i canali colore sopra per identificare il range corretto")
    elif area_traditional > 40:
        print("💡 SUGGERIMENTO: Parametri troppo permissivi")
    else:
        print("✅ Area tradizionale nella range ottimale")

def valuta_solo_immagini_originali():
    """
    Valuta solo le immagini originali, non i risultati
    """
    print("🚀 APPROCCIO IBRIDO BILANCIATO - SOLO IMMAGINI ORIGINALI")
    print("=" * 60)

    # Lista solo delle immagini originali (esclude risultati precedenti)
    immagini_originali = [
        os.path.join(TUE_IMMAGINI_DIR, f)
        for f in os.listdir(TUE_IMMAGINI_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and '_ibrido_result' not in f
        and '_mask' not in f
        and '_overlay' not in f
    ]

    print(f"📸 Trovate {len(immagini_originali)} immagini originali:")
    for img in immagini_originali:
        print(f"   • {os.path.basename(img)}")

    for img_path in immagini_originali:
        print(f"\n" + "="*50)
        print(f"📸 ANALISI: {os.path.basename(img_path)}")
        print("="*50)

        analisi_visuale_dettagliata(img_path)

# === ESECUZIONE ===
print("🎯 ATTIVAZIONE APPROCCIO IBRIDO BILANCIATO")
print("=" * 60)
print("📥 Caricamento modello finale...")
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'combined_loss': combined_loss
    }
)
valuta_solo_immagini_originali()
