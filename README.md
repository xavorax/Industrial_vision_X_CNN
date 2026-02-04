# Reconnaissance de Boîtiers à Boutons Industriels - Vision & CNN

Ce projet permet d'identifier automatiquement la configuration d'un boîtier à boutons industriel (couleurs, nombre, ordre) à partir d'une photo ou d'un flux vidéo en direct. Initialement conçu sous MATLAB, cet algorithme a été porté sous Python en utilisant **OpenCV** pour le traitement d'image et **TensorFlow/CNN** pour l'intelligence artificielle.

## 🚀 Fonctionnalités

* **Localisation Automatique** : Détection du boîtier blanc et extraction de la zone d'intérêt (ROI).
* **Segmentation Avancée** : Identification des boutons via des filtres de surface et de circularité.
* **Data Augmentation** : Génération automatique de variantes d'images (luminosité/saturation) pour renforcer le modèle.
* **Classification par CNN** : Modèle de Deep Learning capable de distinguer 6 classes : `jaune`, `bleu`, `rouge`, `vert`, `aru` (arrêt d'urgence) et `rien` (bruit/vis).
* **Mode Production Real-Time** : Analyse en temps réel via webcam avec affichage des scores de confiance.

## 📁 Structure du Projet

* `analyse_image.py` : Scanne le dossier d'images, extrait les boutons, applique l'augmentation de données et génère le fichier `dataset_complet.csv`.
* `machine_learning.py` : Entraîne le réseau de neurones convolutif (CNN) à partir des vignettes extraites.
* `IA_live.py` : Script final combinant vision par ordinateur et IA pour une détection en direct sur webcam.

## 🛠️ Installation

Assurez-vous d'avoir Python 3.10+ installé. Installez les dépendances nécessaires :

```bash
pip install opencv-python tensorflow pandas numpy scikit-learn

```

## 🧠 Le Modèle CNN

L'architecture utilisée est un modèle séquentiel composé de :

* **3 couches de Convolution (Conv2D)** avec activation ReLU.
* **Couches de MaxPooling2D** pour la réduction de dimension.
* **Une couche Dropout à 0.5** pour éviter le sur-apprentissage (overfitting).
* **Une couche Dense finale** avec activation Softmax pour la classification multi-classe.

## 📋 Utilisation

### 1. Préparation du Dataset

Placez vos photos originales dans le dossier `photo bouton` et lancez l'extraction :

```bash
python analyse_image.py

```

Les vignettes seront créées dans `/extractions` et référencées dans le fichier `dataset_complet.csv`.

### 2. Entraînement

Une fois les données labellisées dans le CSV, lancez l'entraînement :

```bash
python machine_learning.py

```

Le modèle sera sauvegardé sous le nom `modele_boutons_v1.keras`.

### 3. Détection en temps réel

Pour lancer la reconnaissance via la webcam :

```bash
python IA_live.py

```

*Appuyez sur **'Q'** pour quitter le flux vidéo.*

## 📊 Performance & Visualisation

Le système affiche pour chaque bouton détecté :

* Un rectangle englobant (**Bounding Box**).
* Le **label** prédit par l'IA.
* L'indice de **confiance** (en %).
* L'ordre des boutons de gauche à droite dans la console.

## 📈 Évolutions & Pistes d'Amélioration

Bien que performant, le système actuel reste sensible aux variations de luminosité et à la distance focale. Pour stabiliser les performances en environnement industriel, plusieurs axes sont envisageables :

### 1. Vers un Modèle "End-to-End"

Plutôt que de segmenter chaque bouton individuellement, une approche plus robuste consisterait à :
* Utiliser l'algorithme actuel pour **auto-labelliser** un jeu de données massif regroupant tous les formats de boîtiers produits en usine.
* Entraîner un modèle de détection d'objets global (type **YOLO** ou **SSD**) pour reconnaître la boîte complète et sa configuration en une seule passe, minimisant ainsi les erreurs liées au prétraitement d'image.

### 2. Flexibilité du Système Actuel

L'atout majeur de la solution actuelle réside dans sa **modularité** :
* **Adaptabilité rapide** : Le code peut intégrer de nouveaux formats de boîtiers sans nécessiter un réentraînement complet du cœur du modèle.
* **Scalabilité** : L'ajout d'un nouveau type de bouton (couleur ou forme inédite) est extrêmement rapide, nécessitant seulement une courte phase d'extraction et de mise à jour du classifieur CNN.

Pour un accès au donnèes, n'hésitez pas à me contacter :)
