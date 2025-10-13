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
def segmentazione_ibrida_rafforzata(image_path):
    """
    Versione migliorata che rafforza le linee bianche nella maschera ibrida finale
    """
    # Carica immagine
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img_rgb.shape[:2]

    # === FASE 1: PREDIZIONE MODELLO (IDENTICA) ===
    img_processed = cv2.resize(img_rgb, (256, 256))
    img_processed = img_processed.astype(np.float32) / 255.0

    prediction = model.predict(np.expand_dims(img_processed, axis=0), verbose=0)[0].squeeze()
    model_mask = (prediction > 0.15).astype(np.uint8)
    model_mask = cv2.resize(model_mask, (original_w, original_h))

    # === FASE 2: ELABORAZIONE TRADIZIONALE (IDENTICA) ===
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    lower_brown1 = np.array([0, 30, 10])
    upper_brown1 = np.array([30, 220, 150])
    lower_brown2 = np.array([150, 30, 10])
    upper_brown2 = np.array([180, 200, 150])

    mask_hsv1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask_hsv2 = cv2.inRange(hsv, lower_brown2, upper_brown2)

    L, A, B = cv2.split(lab)
    mask_lab_light = cv2.inRange(L, 15, 80)
    mask_lab_color = cv2.inRange(A, 110, 135)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    edges = cv2.Canny(gray_eq, 40, 100)

    kernel = cv2.getGaborKernel((15, 15), 4.0, 0, 10.0, 1.5, 0, ktype=cv2.CV_32F)
    texture_filtered = cv2.filter2D(gray_eq, cv2.CV_8UC3, kernel)
    _, texture_mask = cv2.threshold(texture_filtered, 30, 255, cv2.THRESH_BINARY)

    # Combinazione tradizionale
    color_mask = cv2.bitwise_or(mask_hsv1, mask_hsv2)
    color_mask = cv2.bitwise_or(color_mask, mask_lab_light)
    color_mask = cv2.bitwise_or(color_mask, mask_lab_color)

    structure_mask = cv2.bitwise_or(edges, texture_mask)
    traditional_mask = cv2.bitwise_and(color_mask, structure_mask)

    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    traditional_mask = cv2.morphologyEx(traditional_mask, cv2.MORPH_OPEN, kernel_clean)
    traditional_mask = cv2.morphologyEx(traditional_mask, cv2.MORPH_CLOSE, kernel_clean)

    # === FASE 3: COMBINAZIONE E RAFFORZAMENTO MIGLIORATO ===

    # Prima combinazione base
    final_mask_base = cv2.bitwise_or(traditional_mask, model_mask * 255)

    # === NUOVO: RAFFORZAMENTO SPECIFICO DELLE LINEE ===

    # 1. Estrai le linee sottili dalla maschera tradizionale
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    thin_lines = cv2.morphologyEx(traditional_mask, cv2.MORPH_ERODE, kernel_thin)

    # 2. Dilatazione selettiva per ispessire le linee sottili
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    reinforced_lines = cv2.morphologyEx(thin_lines, cv2.MORPH_DILATE, kernel_dilate)

    # 3. Rilevazione linee con Hough Transform per rami dritti
    lines = cv2.HoughLinesP(traditional_mask, 1, np.pi/180, threshold=15,
                           minLineLength=10, maxLineGap=3)

    lines_mask = np.zeros_like(traditional_mask)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_mask, (x1, y1), (x2, y2), 255, 2)  # Linee più spesse

    # 4. Edge detection più sensibile per rami fini
    edges_fine = cv2.Canny(gray_eq, 20, 50)  # Più sensibile per dettagli fini
    edges_fine = cv2.dilate(edges_fine, kernel_dilate, iterations=1)  # Ispessisce

    # 5. Combina tutti i rinforzi
    all_reinforcements = cv2.bitwise_or(reinforced_lines, lines_mask)
    all_reinforcements = cv2.bitwise_or(all_reinforcements, edges_fine)

    # 6. Applica i rinforzi alla maschera finale
    final_mask = cv2.bitwise_or(final_mask_base, all_reinforcements)

    # 7. Chiusura per connettere linee spezzate
    kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_connect)

    # 8. Pulizia finale conservativa
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)

    # Filtro area ragionevole
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20 or area > (original_h * original_w * 0.4):
            final_mask[labels == i] = 0

    return final_mask, traditional_mask, model_mask * 255

def confronto_rafforzamento(image_path):
    """
    Mostra confronto tra vecchia e nuova versione
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Vecchia versione
    final_mask_old, traditional_old, model_old = segmentazione_ibrida_bilanciata(image_path)
    # Nuova versione
    final_mask_new, traditional_new, model_new = segmentazione_ibrida_rafforzata(image_path)

    # Calcola statistiche
    area_old = np.sum(final_mask_old > 0) / final_mask_old.size * 100
    area_new = np.sum(final_mask_new > 0) / final_mask_new.size * 100
    miglioramento = area_new - area_old

    print(f"🔍 CONFRONTO RAFFORZAMENTO: {os.path.basename(image_path)}")
    print(f"   • Area vecchia: {area_old:.2f}%")
    print(f"   • Area nuova: {area_new:.2f}%")
    print(f"   • Miglioramento: +{miglioramento:.2f}%")

    # Visualizzazione focalizzata sul rafforzamento
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title('Originale')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(model_old, cmap='gray')
    plt.title('Solo U-Net')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(final_mask_old, cmap='gray')
    plt.title(f'Vecchio Ibrido\n{area_old:.1f}%')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(final_mask_new, cmap='gray')
    plt.title(f'Nuovo Ibrido RAFFORZATO\n{area_new:.1f}%')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Overlay comparativo
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.imshow(final_mask_old, cmap='Reds', alpha=0.5)
    plt.title('Overlay Vecchio')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_rgb)
    plt.imshow(final_mask_new, cmap='Reds', alpha=0.5)
    plt.title('Overlay Nuovo (Rafforzato)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    if miglioramento > 0.5:
        print("✅ LINEE MOLTO MIGLIORATE!")
    elif miglioramento > 0.1:
        print("✅ Linee leggermente migliorate")
    else:
        print("ℹ️  Linee simili")

# === TEST DEL RAFFORZAMENTO ===
def test_rafforzamento_linee():
    print("🎯 TEST RAFFORZAMENTO LINEE BIANCHE")
    print("=" * 50)

    immagini_originali = [
        os.path.join(TUE_IMMAGINI_DIR, f)
        for f in os.listdir(TUE_IMMAGINI_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        and '_ibrido_result' not in f
        and '_mask' not in f
        and '_overlay' not in f
    ]

    for img_path in immagini_originali:
        print(f"\n" + "="*40)
        confronto_rafforzamento(img_path)
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
# Esegui il test di rafforzamento
test_rafforzamento_linee()
