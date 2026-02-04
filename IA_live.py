import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
MODEL_PATH = 'modele_boutons_v2.keras'
LABEL_MAP_INV = {0: "jaune", 1: "bleu", 2: "rouge", 3: "aru", 4: "rien", 5:"vert"}
reference_model = ["rouge","vert","bleu"]
# Chargement du modèle
mon_ia = tf.keras.models.load_model(MODEL_PATH)

def preparer_vignette(img_np):
    img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

# --- INITIALISATION WEBCAM ---
cap = cv2.VideoCapture(1) # Essaie 0 ou 1

if not cap.isOpened():
    print("Erreur : Webcam introuvable.")
    exit()

print("Démarrage de l'analyse en temps réel... Appuyez sur 'Q' pour quitter.")
list_color = []
while True:
    buttons_in_frame = []
    ret, frame = cap.read()
    if not ret: break

    # 1. Localisation du boîtier (pour créer la ROI)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bw_box = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    contours_box, _ = cv2.findContours(bw_box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours_box:
        # On prend le plus gros objet blanc
        main_box = max(contours_box, key=cv2.contourArea)
        xb, yb, wb, hb = cv2.boundingRect(main_box)
        
        # Dessiner le contour du boîtier sur le flux
        cv2.rectangle(frame, (xb, yb), (xb + wb, yb + hb), (255, 0, 0), 2)
        cv2.putText(frame, "BOITIER", (xb, yb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Extraction de la zone interne (ROI)
        roi = frame[yb+10 : yb+hb-10, xb+10 : xb+wb-10]
        if roi.size > 0:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 11, 2)
            
            # Détection des boutons dans la ROI
            contours_btn, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours_btn:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                if peri == 0: continue
                circ = (4 * np.pi * area) / (peri**2)

                if 1000 < area and circ > 0.60:
                    x, y, w, h = cv2.boundingRect(cnt)
                    
                    # Crop pour l'IA
                    m = 5
                    crop = roi[max(0, y-m):min(roi.shape[0], y+h+m), 
                               max(0, x-m):min(roi.shape[1], x+w+m)]
                    
                    if crop.size > 0:
                        # Prédiction
                        vignette = preparer_vignette(crop)
                        prediction = mon_ia.predict(vignette, verbose=0)
                        classe_id = np.argmax(prediction)
                        confiance = np.max(prediction)
                        couleur = LABEL_MAP_INV[classe_id]
                        buttons_in_frame.append((y, couleur))

                        # Calcul des coordonnées globales (pour dessiner sur 'frame')
                        # On ajoute les coordonnées de départ du boîtier (xb, yb)
                        global_x = xb + 10 + x
                        global_y = yb + 10 + y

                        # Dessin
                        cv2.rectangle(frame, (global_x, global_y), 
                                        (global_x + w, global_y + h), (0, 255, 0), 2)
                        
                        label_txt = f"{couleur} ({confiance*100:.0f}%)"
                        cv2.putText(frame, label_txt, (global_x, global_y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    buttons_in_frame.sort(key=lambda x: x[0])
    
    # 2. Extraire uniquement les noms de couleurs
    detected_sequence = [b[1] for b in buttons_in_frame]

    # 3. Comparaison avec la référence
    if detected_sequence == reference_model:
        status_text = "CONFORME"
        color_status = (0, 255, 0) # Vert
    else:
        status_text = "NON CONFORME"
        color_status = (0, 0, 255) # Rouge

    # Affichage du statut en haut à droite
    cv2.putText(frame, status_text, (frame.shape[1] - 250, 50), 
                cv2.FONT_HERSHEY_DUPLEX, 1, color_status, 2)
    
    # Optionnel : Afficher la séquence détectée pour débugger
    cv2.putText(frame, f"Seq: {detected_sequence}", (20, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Controle Qualite Boitier', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()