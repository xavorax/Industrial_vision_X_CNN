import cv2
import numpy as np
import csv
import os
import random

# CONFIGURATION-
test = False

if test == True:
    DOSSIER_SOURCE = "photo_test"
    DOSSIER_EXTRACTIONS = "extractions_test"  
    FICHIER_CSV = "dataset_test.csv"
    NB_AUGMENTATIONS = 0
else:
    DOSSIER_SOURCE = "photo_vert"
    DOSSIER_EXTRACTIONS = "extractions_vert"  
    FICHIER_CSV = "Labellisation - dataset_vert.csv"
    NB_AUGMENTATIONS = 3
# Création des dossiers
os.makedirs(DOSSIER_EXTRACTIONS, exist_ok=True)

def modifier_image(img):
    """ Augmentation de données : Saturation et Luminosité. """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= random.uniform(0.6, 1.4) # Saturation
    hsv[:, :, 2] *= random.uniform(0.6, 1.4) # Luminosité
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def analyser_et_extraire(image_path, img_data, csv_writer):
    """ Analyse l'image et sauvegarde une vignette pour chaque bouton trouvé. """
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    _, bw_box = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours_box, _ = cv2.findContours(bw_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours_box: return 0

    main_box = max(contours_box, key=cv2.contourArea)
    xb, yb, wb, hb = cv2.boundingRect(main_box)
    roi = img_data[yb:yb+hb, xb:xb+wb]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours_btn, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    valid_count = 0
    for cnt in contours_btn:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0: continue
    
        circularity = (4 * np.pi * area) / (perimeter**2)

        if 1000 < area and circularity > 0.6:
            valid_count += 1
            
            # Calcul de la couleur moyenne
            mask = np.zeros(roi_gray.shape, dtype="uint8")
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            b, g, r, _ = cv2.mean(roi, mask=mask)
            
            # EXTRACTION DE LA VIGNETTE (CROP)
            x, y, w, h = cv2.boundingRect(cnt)
            # On ajoute une petite marge de 10 pixels pour voir le contour du bouton
            m = 10
            crop = roi[max(0, y-m):min(roi.shape[0], y+h+m), 
                       max(0, x-m):min(roi.shape[1], x+w+m)]
            
            # Sauvegarde du fichier image du bouton
            nom_photo = os.path.basename(image_path).split('.')[0]
            nom_crop = f"{nom_photo}_btn_{valid_count}.jpg"
            chemin_crop = os.path.join(DOSSIER_EXTRACTIONS, nom_crop)
            cv2.imwrite(chemin_crop, crop)
            
            # Enregistrement CSV (avec le chemin du crop en bonus)
            rgb_str = f"({round(r,1)},{round(g,1)},{round(b,1)})"
            csv_writer.writerow(["", image_path, rgb_str, round(area,1), valid_count, chemin_crop])
            
    return valid_count

# BOUCLE PRINCIPALE
images_originales = [f for f in os.listdir(DOSSIER_SOURCE) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and "_modif_" not in f]

with open(FICHIER_CSV, mode='a', newline='') as f:
    writer = csv.writer(f)
    if os.stat(FICHIER_CSV).st_size == 0:
        writer.writerow(["label", "photo_path", "rgb", "surface", "index", "crop_path"])

    for filename in images_originales:
        path_orig = os.path.join(DOSSIER_SOURCE, filename)
        img = cv2.imread(path_orig)
        if img is None: continue
        
        # Analyse originale
        analyser_et_extraire(path_orig, img, writer)

        # Génération des variantes
        for i in range(1, NB_AUGMENTATIONS + 1):
            img_aug = modifier_image(img)
            nom_base, ext = os.path.splitext(filename)
            nom_modif = f"{nom_base}_modif_{i}{ext}"
            path_modif = os.path.join(DOSSIER_SOURCE, nom_modif)
            cv2.imwrite(path_modif, img_aug)
            analyser_et_extraire(path_modif, img_aug, writer)

print(f"Extraction terminée. Vérifie le dossier '{DOSSIER_EXTRACTIONS}' pour voir les boutons isolés.")