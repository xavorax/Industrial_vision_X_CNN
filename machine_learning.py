import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Charger le CSV
df = pd.read_csv("Labelisation - dataset_complet.csv")

# Mapper les labels textuels en nombres
label_map = {"jaune": 0, "bleu": 1, "rouge": 2, "aru": 3, "rien": 4,"vert" : 5}
df['label_num'] = df['label'].map(label_map)

# Séparer en Train Test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Fonction pour charger et préparer l'image
def load_and_preprocess_image(path, label):
    # Lecture du fichier
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    # Redimensionnement à la taille attendue par le modèle (64x64)
    img = tf.image.resize(img, [64, 64])
    # Normalisation (0-1)
    img = img / 255.0
    return img, label

# Création des objets Dataset de TensorFlow
def create_dataset(dataframe, batch_size=32):
    paths = dataframe['crop_path'].values
    labels = dataframe['label_num'].values
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    # Charge les images en parallèle
    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(buffer_size=100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_df)
test_ds = create_dataset(test_df)
"""
def creer_modele_bouton(nb_classes):
    model = models.Sequential([
        # BLOC 1
        layers.Input(shape=(64, 64, 3)), # Taille standardisée
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # BLOC 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # BLOC 3 (Typologie plus fine)
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # CLASSIFICATION
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5), # Évite le sur-apprentissage
        layers.Dense(nb_classes, activation='softmax') # Résultat final
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

mon_ia = creer_modele_bouton(nb_classes=6)

# Entraînement
print("Début de l'entraînement...")
history = mon_ia.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20
)


# 1. Évaluation globale
loss, accuracy = mon_ia.evaluate(test_ds)
print(f"\nPrécision sur le jeu de test : {accuracy*100:.2f}%")

# 2. Sauvegarde du modèle pour l'utiliser plus tard
mon_ia.save('modele_boutons_v2.keras')
print("Modèle sauvegardé sous 'modele_boutons_v1.h5'")
"""
mon_ia = tf.keras.models.load_model('modele_boutons_v1.keras')

# 3. Test de prédiction sur une seule image
def predire_bouton(chemin_image):
    img, _ = load_and_preprocess_image(chemin_image, 0)
    img = tf.expand_dims(img, 0) # Ajouter la dimension "batch"
    prediction = mon_ia.predict(img)
    classe_id = np.argmax(prediction)
    
    # Inverser le dictionnaire pour retrouver le nom
    inv_map = {v: k for k, v in label_map.items()}
    return inv_map[classe_id]

# Exemple :
print(f"Le bouton est : {predire_bouton(r'extractions/WIN_20260123_15_24_22_Pro_btn_5.jpg')}")
