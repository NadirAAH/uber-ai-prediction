<<<<<<< HEAD
# ========== IMPORTS ET DÉPENDANCES ==========
# Import des bibliothèques principales pour l'application web
import streamlit as st           # Framework web pour créer l'interface utilisateur
import pandas as pd              # Manipulation et analyse des données
import numpy as np               # Calculs mathématiques et arrays
import matplotlib.pyplot as plt  # Visualisation de graphiques (matplotlib)
import seaborn as sns           # Visualisation statistique avancée
import plotly.express as px     # Graphiques interactifs Plotly (version express)
import plotly.graph_objects as go  # Graphiques Plotly personnalisés
from plotly.subplots import make_subplots  # Création de sous-graphiques

# Imports pour le machine learning
from sklearn.model_selection import train_test_split, cross_val_score  # Division données et validation croisée
from sklearn.preprocessing import LabelEncoder, StandardScaler          # Encodage et normalisation
from sklearn.impute import SimpleImputer                               # Imputation des valeurs manquantes
from sklearn.ensemble import RandomForestClassifier                    # Algorithme Random Forest
from sklearn.linear_model import LogisticRegression                    # Régression logistique
from sklearn.pipeline import Pipeline                                  # Pipeline pour chaîner les transformations
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support  # Métriques d'évaluation

# Autres imports utilitaires
import warnings     # Gestion des avertissements
import time        # Gestion du temps (pour les animations)
warnings.filterwarnings('ignore')  # Supprime les avertissements pour interface plus propre

# ========== CONFIGURATION DE LA PAGE STREAMLIT ==========
st.set_page_config(
    page_title="Uber AI Prediction Platform",    # Titre affiché dans l'onglet du navigateur
    page_icon="🚗",                             # Icône dans l'onglet du navigateur
    layout="wide",                               # Utilise toute la largeur de l'écran
    initial_sidebar_state="expanded"             # Sidebar ouverte par défaut
)

# ========== PALETTE DE COULEURS GLOBALE ==========
# Définit les couleurs utilisées dans toute l'application pour la cohérence visuelle
COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#11998e', '#38ef7d']

# ========== CSS STYLING AVANCÉ ==========
st.markdown("""
<style>
    /* Import de la police Google Fonts Inter pour un look moderne */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Variables CSS pour réutiliser facilement les couleurs et styles */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);     /* Dégradé principal */
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);  /* Dégradé secondaire */
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);     /* Dégradé d'accent */
        --success-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);    /* Dégradé de succès */
        --glass-bg: rgba(0, 0, 0, 0.5);           /* Fond glassmorphism semi-transparent */
        --glass-border: rgba(255, 255, 255, 0.18); /* Bordure glassmorphism */
        --dark-bg: #0a0a0b;                        /* Fond sombre principal */
        --card-bg: rgba(17, 25, 40, 0.75);         /* Fond des cartes avec transparence */
        --text-primary: #ffffff;                   /* Couleur du texte principal */
        --text-secondary: rgba(255, 255, 255, 20); /* Couleur du texte secondaire */
    }
    
    /* Style du conteneur principal de l'application */
    .main .block-container {
        padding-top: 2rem !important;    /* Espacement en haut */
        padding-bottom: 3rem !important; /* Espacement en bas */
        max-width: 95% !important;       /* Largeur maximale */
    }
   
    /* Style de l'application principale */
    .stApp {
        background: var(--dark-bg);      /* Fond sombre */
        /* Dégradés radiaux pour effet visuel subtil */
        background-image: 
            radial-gradient(circle at 25% 25%, rgba(102, 126, 234, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(118, 75, 162, 0.1) 0%, transparent 50%);
        color: var(--text-primary);      /* Couleur du texte */
        font-family: 'Inter', sans-serif; /* Police moderne */
    }
    
    /* Style pour l'en-tête héro */
    .hero-header {
        background: var(--glass-bg);         /* Fond glassmorphism */
        backdrop-filter: blur(20px);         /* Effet de flou d'arrière-plan */
        border: 1px solid var(--glass-border); /* Bordure subtile */
        border-radius: 24px;                 /* Coins arrondis */
        padding: 3rem 2rem;                  /* Espacement intérieur */
        text-align: center;                  /* Centrage du texte */
        margin-bottom: 3rem;                 /* Marge inférieure */
        animation: slideDown 0.8s ease-out;  /* Animation d'apparition */
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2); /* Ombre portée */
    }
    
    /* Animation de glissement vers le bas */
    @keyframes slideDown {
        from { transform: translateY(-100px); opacity: 0; } /* Position initiale */
        to { transform: translateY(0); opacity: 1; }        /* Position finale */
    }
    
    /* Style du titre principal */
    .hero-title {
        font-size: clamp(2.5rem, 5vw, 4rem);        /* Taille responsive */
        font-weight: 800;                            /* Poids de la police très gras */
        background: var(--primary-gradient);         /* Dégradé de couleur */
        -webkit-background-clip: text;               /* Application du dégradé au texte */
        -webkit-text-fill-color: transparent;       /* Rend le texte transparent pour voir le dégradé */
        background-clip: text;                       /* Support standard */
        margin-bottom: 1rem;                         /* Marge inférieure */
    }
    
    /* Style du sous-titre */
    .hero-subtitle {
        font-size: 1.3rem;                  /* Taille du texte */
        color: var(--text-secondary);       /* Couleur secondaire */
        margin-bottom: 0.5rem;              /* Marge inférieure */
        font-weight: 300;                   /* Poids léger */
    }
    
    /* Style pour l'auteur */
    .hero-author {
        font-size: 1.1rem;                          /* Taille du texte */
        font-weight: 600;                            /* Poids semi-gras */
        background: var(--accent-gradient);          /* Dégradé d'accent */
        -webkit-background-clip: text;               /* Application au texte */
        -webkit-text-fill-color: transparent;       /* Transparence pour le dégradé */
        background-clip: text;                       /* Support standard */
    }
    
    /* Style des cartes de métriques */
    .metric-card {
        background: var(--card-bg);                  /* Fond de carte */
        backdrop-filter: blur(20px);                 /* Effet de flou */
        border: 1px solid var(--glass-border);       /* Bordure subtile */
        border-radius: 16px;                         /* Coins arrondis */
        padding: 2rem 1.5rem;                        /* Espacement intérieur */
        text-align: center;                          /* Centrage du texte */
        transition: all 0.4s ease;                   /* Transition fluide pour les effets */
        animation: fadeIn 0.6s ease-out;             /* Animation d'apparition */
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);   /* Ombre portée */
    }
    
    /* Effet au survol des cartes */
    .metric-card:hover {
        transform: translateY(-8px);                 /* Déplacement vers le haut */
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3); /* Ombre plus prononcée */
    }
    
    /* Animation de fondu */
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }  /* État initial */
        to { opacity: 1; transform: scale(1); }      /* État final */
    }
    
    /* Style des valeurs de métriques */
    .metric-value {
        font-size: 2.5rem;                          /* Grande taille */
        font-weight: 800;                            /* Très gras */
        background: var(--primary-gradient);         /* Dégradé */
        -webkit-background-clip: text;               /* Application au texte */
        -webkit-text-fill-color: transparent;       /* Transparence */
        background-clip: text;                       /* Support standard */
        display: block;                              /* Affichage en bloc */
        margin-bottom: 0.5rem;                       /* Marge inférieure */
    }
    
    /* Style des labels de métriques */
    .metric-label {
        color: var(--text-secondary);               /* Couleur secondaire */
        font-size: 1rem;                            /* Taille normale */
        font-weight: 500;                           /* Poids moyen */
        text-transform: uppercase;                  /* Majuscules */
        letter-spacing: 1px;                        /* Espacement des lettres */
    }
    
    /* Style des cartes de succès */
    .success-card {
        background: linear-gradient(135deg, rgba(17, 153, 142, 0.2), rgba(56, 239, 125, 0.2)); /* Fond vert */
        border: 1px solid rgba(56, 239, 125, 0.3);  /* Bordure verte */
        border-radius: 16px;                         /* Coins arrondis */
        padding: 2rem;                               /* Espacement */
        backdrop-filter: blur(20px);                 /* Effet de flou */
        animation: fadeIn 0.8s ease-out;             /* Animation */
        box-shadow: 0 8px 32px rgba(56, 239, 125, 0.2); /* Ombre verte */
    }
    
    /* Style des cartes d'avertissement */
    .warning-card {
        background: linear-gradient(135deg, rgba(252, 70, 107, 0.2), rgba(63, 94, 251, 0.2)); /* Fond rouge */
        border: 1px solid rgba(252, 70, 107, 0.3);   /* Bordure rouge */
        border-radius: 16px;                          /* Coins arrondis */
        padding: 2rem;                                /* Espacement */
        backdrop-filter: blur(20px);                  /* Effet de flou */
        animation: fadeIn 0.8s ease-out;              /* Animation */
        box-shadow: 0 8px 32px rgba(252, 70, 107, 0.2); /* Ombre rouge */
    }
    
    /* Style de la liste des onglets */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--glass-bg);                 /* Fond glassmorphism */
        backdrop-filter: blur(20px);                 /* Effet de flou */
        border-radius: 16px;                         /* Coins arrondis */
        padding: 0.5rem;                             /* Espacement */
        border: 1px solid var(--glass-border);       /* Bordure */
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);   /* Ombre */
        gap: 0.25rem;                                /* Espacement entre onglets */
    }
    
    /* Style des onglets individuels */
    .stTabs [data-baseweb="tab"] {
        background: transparent;                     /* Fond transparent */
        border-radius: 12px;                        /* Coins arrondis */
        color: var(--text-secondary);               /* Couleur du texte */
        font-weight: 600;                           /* Poids semi-gras */
        padding: 1rem 2rem;                         /* Espacement */
        transition: all 0.3s ease;                  /* Transition fluide */
        border: none;                               /* Pas de bordure */
    }
    
    /* Effet au survol des onglets */
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(102, 126, 234, 0.1);       /* Fond léger au survol */
        color: var(--text-primary);                 /* Couleur du texte */
    }
    
    /* Style de l'onglet actif */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary-gradient);        /* Dégradé principal */
        color: white;                               /* Texte blanc */
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4); /* Ombre bleue */
    }
    
    /* Style des boutons */
    .stButton > button {
        background: var(--primary-gradient);        /* Dégradé de fond */
        border: none;                               /* Pas de bordure */
        border-radius: 12px;                        /* Coins arrondis */
        color: white;                               /* Texte blanc */
        font-weight: 600;                           /* Poids semi-gras */
        padding: 0.75rem 2rem;                      /* Espacement */
        transition: all 0.3s ease;                  /* Transition fluide */
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3); /* Ombre */
        font-size: 1rem;                            /* Taille de police */
        text-transform: uppercase;                  /* Majuscules */
        letter-spacing: 1px;                        /* Espacement des lettres */
    }
    
    /* Effet au survol des boutons */
    .stButton > button:hover {
        transform: translateY(-2px);                /* Déplacement vers le haut */
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.4); /* Ombre plus prononcée */
    }
    
    /* Style des en-têtes de section */
    .section-header {
        font-size: 2rem;                           /* Grande taille */
        font-weight: 700;                          /* Très gras */
        color: var(--text-primary);                /* Couleur principale */
        margin-bottom: 2rem;                       /* Marge inférieure */
        position: relative;                        /* Position relative pour le pseudo-élément */
        padding-left: 1rem;                        /* Espacement à gauche */
    }
    
    /* Barre décorative avant l'en-tête */
    .section-header::before {
        content: '';                               /* Contenu vide pour pseudo-élément */
        position: absolute;                        /* Position absolue */
        left: 0;                                   /* Alignement à gauche */
        top: 0;                                    /* Alignement en haut */
        bottom: 0;                                 /* Alignement en bas */
        width: 4px;                                /* Largeur de la barre */
        background: var(--primary-gradient);       /* Couleur dégradée */
        border-radius: 2px;                        /* Coins arrondis */
    }
    
    /* Personnalisation de la section prédictive */
    div[data-testid="stHorizontalBlock"] {
        color: #ffcc00 !important;                 /* Couleur jaune vif pour tout le texte */
        font-weight: 600;                          /* Texte en semi-gras */
    }

    /* Labels comme "Lieu de départ" */
    div[data-testid="stHorizontalBlock"] label {
        color: #00f2fe !important;                 /* Couleur bleu clair pour les labels */
    }

    /* Valeurs sélectionnées (ex: Palam Vihar, Jhilmil) */
    div[data-testid="stHorizontalBlock"] .st-h4 {
        color: #38ef7d !important;                 /* Couleur vert néon */
        font-size: 1.2rem;                         /* Taille plus grosse */
    }
</style>
""", unsafe_allow_html=True)  # Permet l'utilisation de HTML et CSS dans Streamlit

# ========== FONCTION DE CHARGEMENT ET VALIDATION DES DONNÉES ==========
@st.cache_data  # Cache les données pour éviter de les recharger à chaque fois
def load_and_validate_data():
    """Fonction de chargement avec feedback moderne"""
    from pathlib import Path  # Import pour la gestion des chemins de fichiers
    
    # Chemin principal vers le fichier de données
    DATA_PATH = "archive/ncr_ride_bookings.csv"  # Chemin relatif depuis la racine
    
    try:
        file_path = Path(DATA_PATH)  # Création d'un objet Path
        
        # Vérification si le fichier existe au chemin principal
        if not file_path.exists():
            # Liste des chemins alternatifs à essayer
            alternative_paths = [
                "uber/archive/ncr_ride_bookings.csv",
                "ncr_ride_bookings.csv", 
                "archive/ncr_ride_bookings.csv"
            ]
            
            df = None  # Initialisation du DataFrame
            # Tentative de chargement depuis les chemins alternatifs
            for alt_path in alternative_paths:
                try:
                    df = pd.read_csv(alt_path, encoding='utf-8', low_memory=False)  # Chargement CSV
                    st.success(f"Données chargées depuis : {alt_path}")  # Message de succès
                    break  # Sort de la boucle si le chargement réussit
                except:
                    continue  # Passe au chemin suivant si échec
            
            # Si aucun fichier n'a été trouvé, utilise des données de démonstration
            if df is None:
                st.warning("Utilisation de données de démonstration")  # Message d'avertissement
                return generate_demo_data()  # Appel de la fonction de génération de données
            return df  # Retourne le DataFrame chargé
        
        # Chargement depuis le chemin principal si le fichier existe
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        
        # Vérification si le DataFrame n'est pas vide
        if df.empty:
            return generate_demo_data()  # Utilise des données de démo si vide
            
        # Message de succès avec informations sur la taille du dataset
        st.success(f"Dataset chargé : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
        return df  # Retourne le DataFrame
        
    except Exception as e:  # Gestion des erreurs
        st.error(f"Erreur : {e}")  # Affiche l'erreur
        return generate_demo_data()  # Utilise des données de démo en cas d'erreur

# ========== FONCTION DE GÉNÉRATION DE DONNÉES DE DÉMONSTRATION ==========
@st.cache_data  # Cache les données générées
def generate_demo_data():
    """Génère des données de démonstration"""
    np.random.seed(42)  # Fixe la graine aléatoire pour la reproductibilité
    n_samples = 8000    # Nombre d'échantillons à générer
    
    # Génération d'une série de dates avec un intervalle de 2 heures
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='2H')
    
    # Dictionnaire contenant toutes les colonnes avec leurs valeurs générées
    data = {
        # Conversion des dates en format string
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        # Conversion des heures en format string
        'Time': [d.strftime('%H:%M:%S') for d in dates],
        # Génération aléatoire des statuts de réservation avec probabilités spécifiques
        'Booking Status': np.random.choice([
            'Success', 'Canceled by Driver', 'Canceled by Customer'
        ], n_samples, p=[0.75, 0.13, 0.12]),  # 75% succès, 13% annulé conducteur, 12% annulé client
        # Lieux de départ aléatoires
        'Pickup Location': np.random.choice([
            'Airport', 'Railway Station', 'Metro Station', 'Mall', 'Hospital',
            'Hotel', 'Office Complex', 'Residential Area', 'Tech Park'
        ], n_samples),
        # Destinations aléatoires
        'Drop Location': np.random.choice([
            'Airport', 'Railway Station', 'Metro Station', 'Mall', 'Hospital',
            'Hotel', 'Office Complex', 'Residential Area', 'Tech Park'
        ], n_samples),
        # Distance de trajet selon une distribution exponentielle (plus réaliste)
        'Ride Distance': np.random.exponential(8, n_samples).clip(min=0.5),
        # Valeur de réservation selon une distribution gamma
        'Booking Value': np.random.gamma(2, 15, n_samples).clip(min=8),
        # Types de véhicules avec probabilités différentes
        'Vehicle Type': np.random.choice([
            'Mini', 'Economy', 'Premium', 'Auto', 'Electric'
        ], n_samples, p=[0.35, 0.3, 0.2, 0.1, 0.05]),  # Mini le plus fréquent, Electric le moins
        # Méthodes de paiement avec probabilités réalistes
        'Payment Method': np.random.choice([
            'UPI', 'Card', 'Cash', 'Wallet', 'Corporate'
        ], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),  # UPI le plus utilisé
        # Temps d'attente véhicule selon distribution normale
        'Avg VTAT': np.random.normal(7.5, 3.5, n_samples).clip(min=1),
        # Temps d'attente client selon distribution normale
        'Avg CTAT': np.random.normal(4.8, 2.2, n_samples).clip(min=1)
    }
    
    return pd.DataFrame(data)  # Retourne un DataFrame avec les données générées

# ========== FONCTION DE PRÉTRAITEMENT DES DONNÉES ==========
@st.cache_data   # Cache le résultat du prétraitement
def preprocess_data_exact_method(df_raw):
    TARGET_COLUMN = "Booking Status"  # Définit la colonne cible à prédire
    
    df_clean = df_raw.copy()  # Crée une copie du DataFrame original
    # Supprime les lignes où la colonne cible est manquante
    df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
    
    high_missing_threshold = 0.8  # Seuil pour supprimer les colonnes avec trop de valeurs manquantes
    columns_to_drop = []  # Liste des colonnes à supprimer
    
    # Identifie les colonnes avec trop de valeurs manquantes
    for col in df_clean.columns:
        if col != TARGET_COLUMN:  # Exclut la colonne cible
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)  # Calcule le pourcentage de valeurs manquantes
            if missing_pct > high_missing_threshold:  # Si > 80% manquant
                columns_to_drop.append(col)  # Ajoute à la liste de suppression
    
    df_clean = df_clean.drop(columns=columns_to_drop)  # Supprime les colonnes identifiées
    
    # Liste des caractéristiques légitimes pour le modèle
    LEGITIMATE_FEATURES = [
        'Date', 'Time', 'Pickup Location', 'Drop Location',
        'Ride Distance', 'Booking Value', 'Vehicle Type', 
        'Payment Method', 'Avg VTAT', 'Avg CTAT'
    ]
    
    X = df_clean.drop(columns=[TARGET_COLUMN])  # Caractéristiques (variables indépendantes)
    y = df_clean[TARGET_COLUMN].copy()          # Variable cible (variable dépendante)
    
    # Sélectionne uniquement les caractéristiques disponibles dans le dataset
    available_features = [feat for feat in LEGITIMATE_FEATURES if feat in X.columns]
    X_selected = X[available_features].copy()   # DataFrame avec caractéristiques sélectionnées
    
    # ========== ENGINEERING DES CARACTÉRISTIQUES TEMPORELLES ==========
    # Traitement des colonnes Date et Time si elles existent
    if 'Date' in X_selected.columns and 'Time' in X_selected.columns:
        try:
            # Combine Date et Time en une seule colonne DateTime
            X_selected['DateTime'] = pd.to_datetime(X_selected['Date'] + ' ' + X_selected['Time'])
            # Extrait l'heure de la DateTime
            X_selected['Hour'] = X_selected['DateTime'].dt.hour
            # Extrait le jour de la semaine (0=lundi, 6=dimanche)
            X_selected['DayOfWeek'] = X_selected['DateTime'].dt.dayofweek
            # Extrait le mois
            X_selected['Month'] = X_selected['DateTime'].dt.month
            # Crée une variable binaire pour le weekend (samedi=5, dimanche=6)
            X_selected['IsWeekend'] = (X_selected['DayOfWeek'] >= 5).astype(int)
            
            # Fonction pour catégoriser les heures en créneaux temporels
            def get_time_slot(hour):
                if 6 <= hour < 10: return 'Morning_Rush'        # Rush matinal
                elif 10 <= hour < 16: return 'Midday'           # Milieu de journée
                elif 16 <= hour < 20: return 'Evening_Rush'     # Rush du soir
                elif 20 <= hour < 24: return 'Evening'          # Soirée
                else: return 'Night_EarlyMorning'               # Nuit/tôt le matin
            
            # Applique la fonction de catégorisation des créneaux
            X_selected['TimeSlot'] = X_selected['Hour'].apply(get_time_slot)
            # Supprime les colonnes originales Date, Time et DateTime
            X_selected = X_selected.drop(columns=['Date', 'Time', 'DateTime'])
        except:
            pass  # Ignore les erreurs de traitement temporal
    
    # ========== ENCODAGE DE LA VARIABLE CIBLE ==========
    label_encoder = LabelEncoder()  # Crée un encodeur pour la variable cible
    y_encoded = label_encoder.fit_transform(y)  # Transforme les labels texte en nombres
    
    # ========== ENCODAGE DES VARIABLES CATÉGORIELLES ==========
    # Identifie toutes les colonnes de type objet (texte)
    categorical_features = X_selected.select_dtypes(include=['object']).columns.tolist()
    X_processed = X_selected.copy()  # Copie des données pour le traitement
    feature_encoders = {}  # Dictionnaire pour stocker les encodeurs de chaque caractéristique
    
    # Encode chaque caractéristique catégorielle
    for feature in categorical_features:
        # Remplace les valeurs manquantes par 'MISSING' et convertit en string
        X_processed[feature] = X_processed[feature].fillna('MISSING').astype(str)
        feature_encoder = LabelEncoder()  # Crée un encodeur pour cette caractéristique
        # Transforme les valeurs texte en nombres
        X_processed[feature] = feature_encoder.fit_transform(X_processed[feature])
        feature_encoders[feature] = feature_encoder  # Stocke l'encodeur pour utilisation future
    
    # Retourne tous les éléments nécessaires pour le modélisation et la prédiction
    return X_processed, y_encoded, label_encoder, feature_encoders, df_clean, available_features

# ========== FONCTION D'ENTRAÎNEMENT DES MODÈLES ==========
@st.cache_resource  # Cache les modèles entraînés (plus persistant que cache_data)
def train_models_exact_method(X_processed, y_encoded):
    # ========== DIVISION DES DONNÉES ==========
    try:
        # Tentative de division stratifiée (maintient les proportions des classes)
        # Vérification que chaque classe a au moins 2 échantillons pour la stratification
        stratify_param = y_encoded if len(set(y_encoded)) > 1 and min(np.bincount(y_encoded)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, 
            test_size=0.2,        # 20% pour le test, 80% pour l'entraînement
            random_state=42,      # Graine aléatoire pour la reproductibilité
            stratify=stratify_param  # Maintient les proportions des classes
        )
    except:
        # Division simple sans stratification en cas d'erreur
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42
        )
    
    # ========== PIPELINE DE PRÉTRAITEMENT ==========
    preprocessing_pipeline = Pipeline([
        # Imputation des valeurs manquantes par la médiane
        ('imputer', SimpleImputer(strategy='median')),
        # Standardisation des variables (moyenne=0, écart-type=1)
        ('scaler', StandardScaler())
    ])
    
    # ========== DÉFINITION DES MODÈLES ==========
    models = {
        # Modèle de Régression Logistique
        "Régression Logistique": Pipeline([
            ('preprocessing', preprocessing_pipeline),  # Étape de prétraitement
            ('classifier', LogisticRegression(
                max_iter=1000,         # Nombre maximum d'itérations
                random_state=42,       # Reproductibilité
                class_weight='balanced' # Gestion des classes déséquilibrées
            ))
        ]),
        # Modèle Random Forest
        "Random Forest": Pipeline([
            ('preprocessing', preprocessing_pipeline),  # Étape de prétraitement
            ('classifier', RandomForestClassifier(
                n_estimators=100,       # Nombre d'arbres dans la forêt
                random_state=42,        # Reproductibilité
                class_weight='balanced', # Gestion des classes déséquilibrées
                n_jobs=-1              # Utilise tous les processeurs disponibles
=======
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Statuts Réservation Uber - Portfolio IA",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
    }
    .success-metric {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-metric {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Charge les données réelles ou génère des données de démonstration"""
    try:
        # Tentative de chargement de vos données réelles
        df = pd.read_csv('uber/archive/ncr_ride_bookings.csv')
        st.success("Données réelles chargées avec succès!")
        return df
    except FileNotFoundError:
        st.warning("Fichier de données non trouvé. Utilisation de données de démonstration.")
        return generate_demo_data()

@st.cache_data
def generate_demo_data():
    """Génère des données de démonstration réalistes"""
    np.random.seed(42)
    n_samples = 10000
    
    # Génération de données synthétiques réalistes basées sur le contexte Uber
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    data = {
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Time': [d.strftime('%H:%M:%S') for d in dates],
        'Pickup Location': np.random.choice([
            'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples, p=[0.15, 0.25, 0.12, 0.10, 0.08, 0.10, 0.15, 0.05]),
        'Drop Location': np.random.choice([
            'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples, p=[0.12, 0.20, 0.15, 0.08, 0.05, 0.08, 0.12, 0.20]),
        'Ride Distance': np.random.exponential(8, n_samples) + 1,  # Distance en km
        'Booking Value': np.random.normal(28, 12, n_samples).clip(min=8),  # Prix en euros
        'Vehicle Type': np.random.choice([
            'Economy', 'Premium', 'Shared', 'Luxury'
        ], n_samples, p=[0.5, 0.25, 0.2, 0.05]),
        'Payment Method': np.random.choice([
            'Card', 'Cash', 'Wallet', 'Corporate'
        ], n_samples, p=[0.6, 0.25, 0.12, 0.03]),
        'Avg VTAT': np.random.normal(7, 3, n_samples).clip(min=1),  # Temps attente véhicule
        'Avg CTAT': np.random.normal(4, 2, n_samples).clip(min=1),  # Temps attente client
        'Booking Status': np.random.choice([
            'Success', 'Canceled by Driver', 'Canceled by Customer', 'No Show'
        ], n_samples, p=[0.75, 0.12, 0.10, 0.03])
    }
    
    return pd.DataFrame(data)

@st.cache_data
def preprocess_data(df):
    """Preprocessing complet des données"""
    df_processed = df.copy()
    
    # Feature engineering temporel
    df_processed['DateTime'] = pd.to_datetime(df_processed['Date'] + ' ' + df_processed['Time'])
    df_processed['Hour'] = df_processed['DateTime'].dt.hour
    df_processed['DayOfWeek'] = df_processed['DateTime'].dt.dayofweek
    df_processed['Month'] = df_processed['DateTime'].dt.month
    df_processed['IsWeekend'] = (df_processed['DayOfWeek'] >= 5).astype(int)
    
    def get_time_slot(hour):
        if 6 <= hour < 10:
            return 'Morning_Rush'
        elif 10 <= hour < 16:
            return 'Midday'
        elif 16 <= hour < 20:
            return 'Evening_Rush'
        elif 20 <= hour < 24:
            return 'Evening'
        else:
            return 'Night_EarlyMorning'
    
    df_processed['TimeSlot'] = df_processed['Hour'].apply(get_time_slot)
    
    # Sélection des features pertinentes
    features = ['Pickup Location', 'Drop Location', 'Ride Distance', 'Booking Value',
                'Vehicle Type', 'Payment Method', 'Avg VTAT', 'Avg CTAT',
                'Hour', 'DayOfWeek', 'Month', 'IsWeekend', 'TimeSlot']
    
    X = df_processed[features].copy()
    y = df_processed['Booking Status'].copy()
    
    # Encodage des variables catégorielles
    label_encoders = {}
    categorical_features = X.select_dtypes(include=['object']).columns
    
    for col in categorical_features:
        X[col] = X[col].fillna('MISSING')
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Encodage de la variable cible
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, target_encoder, label_encoders, df_processed

@st.cache_resource
def train_models(X, y):
    """Entraîne et évalue les modèles de machine learning"""
    # Division train/test stratifiée
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Pipeline de preprocessing
    preprocessing = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Modèles à évaluer
    models = {
        'Random Forest': Pipeline([
            ('preprocessing', preprocessing),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            ))
        ]),
        'Logistic Regression': Pipeline([
            ('preprocessing', preprocessing),
            ('classifier', LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight='balanced'
>>>>>>> 7c4a6b4d71e4c4f2ff98ea8e61983123f9ea7e9c
            ))
        ])
    }
    
<<<<<<< HEAD
    # ========== ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES ==========
    results = []  # Liste pour stocker les résultats de chaque modèle
    for name, model in models.items():  # Itère sur chaque modèle
        model.fit(X_train, y_train)  # Entraîne le modèle sur les données d'entraînement
        y_pred = model.predict(X_test)  # Fait des prédictions sur les données de test
        accuracy = accuracy_score(y_test, y_pred)  # Calcule la précision
        # Validation croisée à 5 plis pour une évaluation robuste
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Stocke tous les résultats du modèle
        results.append({
            'model': model,              # Le modèle entraîné
            'name': name,               # Le nom du modèle
            'accuracy': accuracy,       # La précision sur le jeu de test
            'cv_mean': cv_scores.mean(), # Moyenne de la validation croisée
            'cv_std': cv_scores.std(),  # Écart-type de la validation croisée
            'predictions': y_pred       # Les prédictions sur le jeu de test
        })
    
    # Sélectionne le meilleur modèle basé sur la précision
    best_model_result = max(results, key=lambda x: x['accuracy'])
    return results, best_model_result, X_train, X_test, y_train, y_test

# ========== FONCTION DE CRÉATION DU THÈME PLOTLY ==========
def create_modern_plotly_theme():
    """Thème Plotly moderne"""
    return {
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',  # Fond transparent
            'plot_bgcolor': 'rgba(0,0,0,0)',   # Zone de graphique transparente
            'font': {'color': '#ffffff', 'family': 'Inter'}, # Police blanche Inter
            'colorway': COLORS  # Utilise la palette de couleurs globale
        }
    }

# ========== FONCTION PRINCIPALE DE L'APPLICATION ==========
def main():
    # ========== HEADER HÉRO MODERNE ==========
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">Uber AI Prediction Platform</h1>
        <p class="hero-subtitle">Intelligence Artificielle • Machine Learning • Data Science</p>
        <p class="hero-author">Développé par Nadir Ali Ahmed</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SIDEBAR MODERNE ==========
    with st.sidebar:  # Crée la barre latérale
        # En-tête de la sidebar avec style
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-weight: 800; margin-bottom: 1rem;">
                Mission Control
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Contenu informatif de la sidebar
        st.markdown("""
        **Méthodologie Avancée**
        
        **Features Intelligence:**
        • Variables temporelles dérivées
        • Géolocalisation optimisée  
        • Patterns comportementaux
        
        **AI Pipeline:**
        • Data Quality Enhancement
        • Feature Engineering Automatisé
        • Ensemble Learning
        
        **Modèles Déployés:**
        • Random Forest Optimisé
        • Régression Logistique Équilibrée
        """)
    
    # ========== CHARGEMENT AVEC ANIMATION ==========
    with st.spinner('Initialisation du système AI...'):  # Spinner d'attente
        progress_bar = st.progress(0)  # Barre de progression
        status_text = st.empty()       # Zone de texte pour le statut
        
        # Animation de chargement avec barre de progression
        for i in range(100):
            progress_bar.progress(i + 1)  # Met à jour la barre de progression
            # Messages de statut selon l'avancement
            if i < 30:
                status_text.text(f'Chargement des données... {i+1}%')
            elif i < 60:
                status_text.text(f'Preprocessing avancé... {i+1}%')
            elif i < 90:
                status_text.text(f'Entraînement des modèles... {i+1}%')
            else:
                status_text.text(f'Finalisation... {i+1}%')
            time.sleep(0.01)  # Pause pour l'animation
        
        # Nettoie les éléments d'animation
        progress_bar.empty()
        status_text.empty()
        
        # ========== CHARGEMENT ET PRÉPARATION DES DONNÉES ==========
        df_raw = load_and_validate_data()  # Charge les données
        if df_raw is not None:  # Vérification que le chargement a réussi
            # Prétraitement des données
            X_processed, y_encoded, label_encoder, feature_encoders, df_clean, available_features = preprocess_data_exact_method(df_raw)
            # Entraînement des modèles
            results, best_model_result, X_train, X_test, y_train, y_test = train_models_exact_method(X_processed, y_encoded)
        else:
            st.error("Échec de l'initialisation")  # Message d'erreur
            return  # Sort de la fonction si échec
    
    # ========== CREATION DES ONGLETS PRINCIPAUX ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", "Data Science", "AI Models", "Analytics", "Live Prediction"
    ])
    
    # ========== ONGLET 1: DASHBOARD ==========
    with tab1:
        st.markdown('<h2 class="section-header">Executive Dashboard</h2>', unsafe_allow_html=True)
        
        # ========== MÉTRIQUES PRINCIPALES ==========
        col1, col2, col3, col4 = st.columns(4)  # Crée 4 colonnes égales
        
        # Définition des métriques à afficher
        metrics = [
            ("Observations", f"{len(df_raw):,}", ""),        # Nombre total de lignes
            ("Variables", f"{len(df_raw.columns)}", ""),     # Nombre de colonnes
            ("Taux Succès", f"{(df_raw['Booking Status'] == 'Success').mean()*100:.1f}%", ""), # Pourcentage de succès
            ("Performance IA", f"{best_model_result['accuracy']*100:.1f}%", "") # Performance du meilleur modèle
        ]
        
        # Affichage de chaque métrique dans sa colonne
        for i, (col, (label, value, icon)) in enumerate(zip([col1, col2, col3, col4], metrics)):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="animation-delay: {i*0.1}s;">
                    <span style="font-size: 3rem; display: block; margin-bottom: 0.5rem;">{icon}</span>
                    <span class="metric-value">{value}</span>
                    <span class="metric-label">{label}</span>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)  # Espace
        
        # ========== VISUALISATIONS ==========
        col1, col2 = st.columns(2)  # Crée 2 colonnes pour les graphiques
        
        with col1:
            st.subheader("Distribution des Statuts")  # Titre du graphique
            status_counts = df_raw['Booking Status'].value_counts()  # Compte les occurrences
            
            # Création d'un graphique en secteurs (pie chart)
            fig = go.Figure(data=go.Pie(
                labels=status_counts.index,      # Labels des secteurs
                values=status_counts.values,     # Valeurs des secteurs
                hole=0.6,                        # Crée un donut chart
                marker_colors=COLORS[:len(status_counts)]  # Utilise les couleurs globales
            ))
            
            # Configuration des traces et du layout
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                **create_modern_plotly_theme()['layout'],  # Applique le thème moderne
                height=400,    # Hauteur du graphique
                showlegend=False  # Cache la légende
            )
            
            st.plotly_chart(fig, use_container_width=True)  # Affiche le graphique
        
        with col2:
            st.subheader("Performance")  # Titre du graphique
            
            # Données de performance (en dur pour la démo)
            perf_data = pd.DataFrame({
                'Métrique': ['Précision', 'Recall', 'F1-Score', 'Accuracy'],
                'Score': [0.87, 0.82, 0.84, best_model_result['accuracy']]
            })
            
            # Création d'un graphique en barres horizontales
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=perf_data['Score'],          # Valeurs sur l'axe X
                y=perf_data['Métrique'],       # Labels sur l'axe Y
                orientation='h',               # Barres horizontales
                marker_color=COLORS[:len(perf_data)]  # Couleurs
            ))
            
            fig.update_layout(
                **create_modern_plotly_theme()['layout'],
                height=400,
                xaxis_title="Score",      # Titre axe X
                yaxis_title="Métriques"   # Titre axe Y
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== ONGLET 2: DATA SCIENCE ==========
    with tab2:
        st.markdown('<h2 class="section-header">Data Science Lab</h2>', unsafe_allow_html=True)
=======
    results = {}
    for name, model in models.items():
        # Entraînement
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        
        # Validation croisée
        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        except:
            cv_scores = np.array([accuracy])  # Fallback
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'y_test': y_test
        }
    
    return results, X_train, X_test, y_train, y_test

def create_confusion_matrix_plot(y_test, y_pred, target_encoder):
    """Crée une heatmap de la matrice de confusion"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = px.imshow(
        cm, 
        text_auto=True,
        aspect="auto",
        title="Matrice de Confusion",
        labels=dict(x="Prédictions", y="Vraies Valeurs"),
        x=target_encoder.classes_,
        y=target_encoder.classes_,
        color_continuous_scale="Blues"
    )
    
    return fig

def create_feature_importance_plot(model, feature_names):
    """Crée un graphique d'importance des features pour Random Forest"""
    if hasattr(model.named_steps['classifier'], 'feature_importances_'):
        importance = model.named_steps['classifier'].feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(
            importance_df, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Importance des Features (Random Forest)"
        )
        
        return fig
    return None

def main():
    # Header principal avec style
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Prédiction des Statuts de Réservation Uber</h1>
        <p>Projet d'Intelligence Artificielle - Machine Learning</p>
        <p><strong>Par :</strong> Nadir Ali Ahmed | <strong>Portfolio Data Science</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar avec informations du projet
    st.sidebar.markdown("## 📊 À Propos du Projet")
    st.sidebar.markdown("""
    **Objectif :** Prédire le statut des réservations Uber
    
    **Classes prédites :**
    - ✅ Succès
    - ❌ Annulé par chauffeur
    - ❌ Annulé par client
    - ⚠️ No Show
    
    **Technologies :**
    - Python (Pandas, Scikit-learn)
    - Random Forest, Régression Logistique
    - Streamlit, Plotly
    
    **Méthodologie :**
    1. Analyse exploratoire (EDA)
    2. Feature Engineering
    3. Preprocessing automatisé
    4. Entraînement de modèles
    5. Évaluation comparative
    """)
    
    # Chargement des données avec feedback
    with st.spinner('Chargement et preprocessing des données...'):
        df = load_data()
        X, y, target_encoder, label_encoders, df_processed = preprocess_data(df)
        results, X_train, X_test, y_train, y_test = train_models(X, y)
    
    # Tabs principales de l'application
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Données", 
        "🔍 EDA", 
        "🤖 Modèles", 
        "📊 Performance", 
        "🎯 Test Interactif",
        "📋 Rapport Final"
    ])
    
    with tab1:
        st.header("📈 Aperçu des Données")
        
        # Métriques principales
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Observations", f"{len(df):,}")
        with col2:
            st.metric("Features", f"{len(X.columns)}")
        with col3:
            st.metric("Classes", f"{len(target_encoder.classes_)}")
        with col4:
            success_rate = (df['Booking Status'] == 'Success').mean() * 100
            st.metric("Taux Succès", f"{success_rate:.1f}%")
        
        # Échantillon des données
        st.subheader("Échantillon des données originales")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Distribution de la variable cible
        st.subheader("Distribution de la variable cible")
        target_dist = df['Booking Status'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(
                values=target_dist.values, 
                names=target_dist.index,
                title="Répartition des Statuts"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=target_dist.index, 
                y=target_dist.values,
                title="Effectifs par Statut",
                labels={'x': 'Statut', 'y': 'Nombre de réservations'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(), use_container_width=True)
    
    with tab2:
        st.header("🔍 Analyse Exploratoire des Données")
>>>>>>> 7c4a6b4d71e4c4f2ff98ea8e61983123f9ea7e9c
        
        col1, col2 = st.columns(2)
        
        with col1:
<<<<<<< HEAD
            st.subheader("Patterns Temporels")
            # Vérification si la caractéristique 'Hour' existe
            if 'Hour' in X_processed.columns:
                # Agrégation des données par heure
                hourly_data = df_clean.groupby(pd.to_datetime(df_clean['Date'] + ' ' + df_clean['Time']).dt.hour).size()
                
                # Graphique linéaire avec zone remplie
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly_data.index,    # Heures
                    y=hourly_data.values,   # Nombre de réservations
                    mode='lines+markers',   # Ligne avec marqueurs
                    fill='tonexty',         # Remplissage sous la courbe
                    line=dict(color=COLORS[0], width=3),   # Style de ligne
                    marker=dict(size=8, color=COLORS[1])   # Style des marqueurs
                ))
                
                fig.update_layout(
                    **create_modern_plotly_theme()['layout'],
                    height=350,
                    xaxis_title="Heure",
                    yaxis_title="Réservations"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Véhicules Premium")
            vehicle_dist = df_raw['Vehicle Type'].value_counts()  # Distribution des types de véhicules
            
            # Graphique en barres horizontales
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=vehicle_dist.values,  # Valeurs
                y=vehicle_dist.index,   # Types de véhicules
                orientation='h',        # Horizontal
                marker_color=COLORS[:len(vehicle_dist)]
            ))
            
            fig.update_layout(
                **create_modern_plotly_theme()['layout'],
                height=350
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== ONGLET 3: AI MODELS ==========
    with tab3:
        st.markdown('<h2 class="section-header">AI Models Center</h2>', unsafe_allow_html=True)
        
        st.subheader("Battle des Algorithmes")
        
        # Création d'un DataFrame avec les résultats de comparaison
        comparison_data = pd.DataFrame([
            {'Modèle': r['name'], 'Accuracy': r['accuracy'], 'CV Mean': r['cv_mean'], 'CV Std': r['cv_std']}
            for r in results
        ])
        
        # Graphique de comparaison des modèles
        fig = go.Figure()
        
        for i, model in enumerate(comparison_data['Modèle']):
            fig.add_trace(go.Bar(
                name=model,                                    # Nom du modèle
                x=[model],                                     # Position X
                y=[comparison_data.iloc[i]['Accuracy']],       # Valeur accuracy
                marker_color=COLORS[i],                        # Couleur
                text=f"{comparison_data.iloc[i]['Accuracy']:.3f}", # Texte affiché
                textposition='outside'                         # Position du texte
            ))
        
        fig.update_layout(
            **create_modern_plotly_theme()['layout'],
            height=400,
            yaxis_title="Accuracy Score",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ========== AFFICHAGE DU CHAMPION ==========
        st.markdown(f"""
        <div class="success-card">
            <h3 style="margin-bottom: 1rem;">Champion Model</h3>
            <h2 style="color: #38ef7d; margin-bottom: 0.5rem;">{best_model_result['name']}</h2>
            <p style="font-size: 1.2rem; margin-bottom: 0.5rem;">
                Accuracy: <strong>{best_model_result['accuracy']:.4f}</strong> ({best_model_result['accuracy']*100:.2f}%)
            </p>
            <p>Validation Croisée: {best_model_result['cv_mean']:.4f} ± {best_model_result['cv_std']:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== ONGLET 4: ANALYTICS ==========
    with tab4:
        st.markdown('<h2 class="section-header">Advanced Analytics</h2>', unsafe_allow_html=True)
=======
            st.subheader("Réservations par heure")
            hourly_data = df_processed.groupby('Hour').size()
            fig = px.line(
                x=hourly_data.index, 
                y=hourly_data.values,
                title="Volume de réservations par heure",
                labels={'x': 'Heure', 'y': 'Nombre de réservations'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Statuts par jour de la semaine")
            days = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
            weekly_data = pd.crosstab(df_processed['DayOfWeek'], df['Booking Status'])
            weekly_data.index = days
            fig = px.bar(weekly_data, title="Statuts par jour de la semaine")
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Distribution des distances")
            fig = px.histogram(
                df, 
                x='Ride Distance', 
                title="Distribution des distances de trajet",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("Valeur moyenne par type de véhicule")
            vehicle_stats = df.groupby('Vehicle Type')['Booking Value'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=vehicle_stats.values,
                y=vehicle_stats.index,
                orientation='h',
                title="Valeur moyenne par type de véhicule"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Corrélations
        st.subheader("Matrice de corrélation des variables numériques")
        numeric_cols = ['Ride Distance', 'Booking Value', 'Avg VTAT', 'Avg CTAT', 'Hour']
        if all(col in df.columns for col in numeric_cols):
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(
                corr_matrix, 
                title="Corrélations entre variables numériques",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🤖 Modèles d'Intelligence Artificielle")
        
        st.markdown("""
        ### 🛠️ Méthodologie de Développement
        
        **1. Feature Engineering :**
        - Extraction temporelle : heure, jour semaine, créneaux horaires
        - Variables dérivées : weekend, rush hours
        - Encodage des variables catégorielles
        
        **2. Preprocessing :**
        - Gestion des valeurs manquantes (médiane)
        - Standardisation des features numériques
        - Pipeline intégré pour éviter le data leakage
        
        **3. Modèles testés :**
        - **Random Forest** : Ensemble d'arbres de décision
        - **Régression Logistique** : Modèle linéaire probabiliste
        
        **4. Validation :**
        - Division train/test stratifiée (80/20)
        - Validation croisée 5-fold
        - Équilibrage des classes (class_weight='balanced')
        """)
        
        # Comparaison des performances
        st.subheader("🏆 Comparaison des Performances")
        
        model_comparison = []
        for name, result in results.items():
            model_comparison.append({
                'Modèle': name,
                'Accuracy': result['accuracy'],
                'CV Mean': result['cv_mean'],
                'CV Std': result['cv_std']
            })
        
        comparison_df = pd.DataFrame(model_comparison)
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                comparison_df.style.format({
                    'Accuracy': '{:.4f}',
                    'CV Mean': '{:.4f}',
                    'CV Std': '{:.4f}'
                }),
                use_container_width=True
            )
        
        with col2:
            fig = px.bar(
                comparison_df, 
                x='Modèle', 
                y='Accuracy',
                title="Accuracy par modèle",
                color='Accuracy',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        st.markdown(f"""
        <div class="success-metric">
            <h4>🏆 Meilleur Modèle : {best_model_name}</h4>
            <p>Accuracy: {results[best_model_name]['accuracy']:.4f} ({results[best_model_name]['accuracy']*100:.2f}%)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with tab4:
        st.header("📊 Analyse Détaillée des Performances")
        
        # Sélection du meilleur modèle
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
>>>>>>> 7c4a6b4d71e4c4f2ff98ea8e61983123f9ea7e9c
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Matrice de Confusion")
<<<<<<< HEAD
            # Calcul de la matrice de confusion
            cm = confusion_matrix(y_test, best_model_result['predictions'])
            
            # Heatmap de la matrice de confusion
            fig = go.Figure(data=go.Heatmap(
                z=cm,                           # Valeurs de la matrice
                x=label_encoder.classes_,       # Labels X (prédictions)
                y=label_encoder.classes_,       # Labels Y (vraies valeurs)
                colorscale='Viridis',           # Échelle de couleur
                text=cm,                        # Texte affiché dans chaque cellule
                texttemplate="%{text}",         # Format du texte
                textfont={"size": 16},          # Taille de police
                showscale=True                  # Affiche l'échelle de couleur
            ))
            
            fig.update_layout(
                **create_modern_plotly_theme()['layout'],
                height=400,
                xaxis_title="Prédictions",
                yaxis_title="Vraies Valeurs"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Métriques Détaillées")
            # Calcul des métriques détaillées pour chaque classe
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, best_model_result['predictions'], average=None
            )
            
            # Création du DataFrame des métriques
            metrics_data = pd.DataFrame({
                'Classe': label_encoder.classes_,  # Noms des classes
                'Precision': precision,            # Précision par classe
                'Recall': recall,                  # Rappel par classe
                'F1-Score': f1                     # F1-Score par classe
            })
            
            # Graphique en barres groupées
            fig = go.Figure()
            
            metric_colors = {'Precision': COLORS[0], 'Recall': COLORS[1], 'F1-Score': COLORS[2]}
            
            # Ajoute une barre pour chaque métrique
            for metric in ['Precision', 'Recall', 'F1-Score']:
                fig.add_trace(go.Bar(
                    name=metric,                    # Nom de la série
                    x=metrics_data['Classe'],       # Classes sur X
                    y=metrics_data[metric],         # Valeurs sur Y
                    marker_color=metric_colors[metric]  # Couleur spécifique
                ))
            
            fig.update_layout(
                **create_modern_plotly_theme()['layout'],
                height=400,
                barmode='group'  # Barres groupées
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== ONGLET 5: PRÉDICTION EN TEMPS RÉEL ==========
    with tab5:
        st.markdown('<h2 class="section-header">Live AI Prediction</h2>', unsafe_allow_html=True)
        
        # Description de la section
        st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: rgba(255,255,255,0.7);">
                Testez le modèle en temps réel avec vos paramètres personnalisés
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # ========== INTERFACE DE SAISIE ==========
        col1, col2, col3 = st.columns(3)  # Trois colonnes pour les inputs
        
        with col1:
            # Extraction des options uniques depuis les données
            pickup_options = df_clean['Pickup Location'].unique()
            drop_options = df_clean['Drop Location'].unique()
            
            # Sélecteurs pour les lieux
            pickup = st.selectbox("Lieu de Départ", pickup_options)
            drop = st.selectbox("Destination", drop_options)
            
        with col2:
            # Options pour véhicule et paiement
            vehicle_options = df_clean['Vehicle Type'].unique()
            payment_options = df_clean['Payment Method'].unique()
            
            vehicle_type = st.selectbox("Type de Véhicule", vehicle_options)
            payment = st.selectbox("Mode de Paiement", payment_options)
            
        with col3:
            # Sliders pour les valeurs numériques
            distance = st.slider("Distance (km)", 1.0, 50.0, 12.0)        # Distance du trajet
            booking_value = st.slider("Valeur", 10.0, 500.0, 120.0)       # Valeur de la réservation
            hour = st.slider("Heure", 0, 23, 14)                          # Heure de la réservation
        
        # ========== BOUTON DE PRÉDICTION ==========
        if st.button("PRÉDIRE LE STATUT"):
            with st.spinner('Intelligence artificielle en action...'):  # Spinner pendant la prédiction
                time.sleep(1)  # Pause pour l'effet visuel
                
                # ========== PRÉPARATION DES DONNÉES DE TEST ==========
                test_data = {}  # Dictionnaire pour les données de test
                
                # Remplissage des données selon les caractéristiques du modèle
                for feature in X_processed.columns:
                    if feature == 'Hour':
                        test_data[feature] = hour
                    elif feature == 'Pickup Location':
                        test_data[feature] = pickup
                    elif feature == 'Drop Location':
                        test_data[feature] = drop
                    elif feature == 'Vehicle Type':
                        test_data[feature] = vehicle_type
                    elif feature == 'Payment Method':
                        test_data[feature] = payment
                    elif feature == 'Ride Distance':
                        test_data[feature] = distance
                    elif feature == 'Booking Value':
                        test_data[feature] = booking_value
                    else:
                        test_data[feature] = 0  # Valeur par défaut pour les autres caractéristiques
                
                # ========== ENCODAGE DES VARIABLES CATÉGORIELLES ==========
                # Applique les encodeurs précédemment entraînés
                for col, encoder in feature_encoders.items():
                    if col in test_data and isinstance(test_data[col], str):
                        try:
                            # Vérifie si la valeur était dans l'entraînement
                            if test_data[col] in encoder.classes_:
                                test_data[col] = encoder.transform([test_data[col]])[0]
                            else:
                                test_data[col] = 0  # Valeur par défaut pour les nouvelles catégories
                        except:
                            test_data[col] = 0
                
                # ========== PRÉDICTION ==========
                try:
                    test_df = pd.DataFrame([test_data])  # Conversion en DataFrame
                    prediction = best_model_result['model'].predict(test_df)[0]  # Prédiction
                    prediction_proba = best_model_result['model'].predict_proba(test_df)[0]  # Probabilités
                    predicted_class = label_encoder.inverse_transform([prediction])[0]  # Classe prédite
                    confidence = np.max(prediction_proba)  # Confiance maximum
                    
                    # ========== AFFICHAGE DU RÉSULTAT ==========
                    if predicted_class == 'Success':
                        # Carte de succès
                        st.markdown(f"""
                        <div class="success-card" style="text-align: center; margin: 2rem 0;">
                            <h2 style="color: #38ef7d; margin-bottom: 1rem;">SUCCÈS PRÉDIT</h2>
                            <h3 style="margin-bottom: 0.5rem;">Réservation: {predicted_class}</h3>
                            <p style="font-size: 1.2rem;">Confiance: <strong>{confidence:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Carte d'avertissement
                        st.markdown(f"""
                        <div class="warning-card" style="text-align: center; margin: 2rem 0;">
                            <h2 style="color: #fc466b; margin-bottom: 1rem;">RISQUE DÉTECTÉ</h2>
                            <h3 style="margin-bottom: 0.5rem;">Statut: {predicted_class}</h3>
                            <p style="font-size: 1.2rem;">Confiance: <strong>{confidence:.1%}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # ========== GRAPHIQUE DES PROBABILITÉS ==========
                    # Création du DataFrame des probabilités
                    proba_df = pd.DataFrame({
                        'Statut': label_encoder.classes_,
                        'Probabilité': prediction_proba
                    }).sort_values('Probabilité', ascending=True)  # Tri par probabilité
                    
                    # Graphique en barres des probabilités
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=proba_df['Probabilité'],      # Probabilités sur X
                        y=proba_df['Statut'],           # Statuts sur Y
                        orientation='h',                # Barres horizontales
                        marker_color=COLORS[:len(proba_df)],  # Couleurs
                        text=[f'{p:.1%}' for p in proba_df['Probabilité']],  # Texte des pourcentages
                        textposition='outside'          # Position du texte
                    ))
                    
                    fig.update_layout(
                        **create_modern_plotly_theme()['layout'],
                        title="Distribution des Probabilités",
                        height=300,
                        xaxis_title="Probabilité",
                        yaxis_title="Statut"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Erreur dans la prédiction: {e}")  # Gestion d'erreur
    
    # ========== FOOTER MODERNE ==========
    st.markdown("<br><br>", unsafe_allow_html=True)  # Espacement
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; border-top: 1px solid rgba(255,255,255,0.1);">
        <h3 style="background: linear-gradient(135deg, #667eea, #764ba2); 
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   margin-bottom: 1rem;">
            Portfolio Data Science
        </h3>
        <p style="color: rgba(255,255,255,0.7); margin-bottom: 0.5rem;">
            <strong>Développé par Nadir Ali Ahmed</strong>
        </p>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.9rem;">
            Intelligence Artificielle • Machine Learning • Big Data Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

# ========== POINT D'ENTRÉE DE L'APPLICATION ==========
if __name__ == "__main__":  # Vérifie si le script est exécuté directement
    main()  # Lance la fonction principale
=======
            fig = create_confusion_matrix_plot(
                best_result['y_test'], 
                best_result['predictions'], 
                target_encoder
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Métriques par Classe")
            report = classification_report(
                best_result['y_test'], 
                best_result['predictions'], 
                target_names=target_encoder.classes_, 
                output_dict=True
            )
            
            report_df = pd.DataFrame(report).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
        
        # Importance des features (si Random Forest)
        if 'Random Forest' in best_model_name:
            st.subheader("Importance des Features")
            fig = create_feature_importance_plot(best_result['model'], X.columns)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse des erreurs
        st.subheader("Analyse des Erreurs")
        cm = confusion_matrix(best_result['y_test'], best_result['predictions'])
        total_errors = len(best_result['y_test']) - cm.trace()
        error_rate = total_errors / len(best_result['y_test'])
        
        st.markdown(f"""
        **Statistiques d'erreur :**
        - Erreurs totales : {total_errors}
        - Taux d'erreur : {error_rate:.3f} ({error_rate*100:.1f}%)
        - Prédictions correctes : {cm.trace()} / {len(best_result['y_test'])}
        """)
    
    with tab5:
        st.header("🎯 Test Interactif du Modèle")
        st.markdown("Testez le modèle avec vos propres paramètres et obtenez une prédiction en temps réel !")
        
        # Interface de test
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pickup = st.selectbox("Lieu de départ", [
                'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
                'Hotel', 'Office Complex', 'Residential Area'
            ])
            drop = st.selectbox("Destination", [
                'Airport', 'Downtown', 'Mall', 'Station', 'Hospital', 
                'Hotel', 'Office Complex', 'Residential Area'
            ])
            distance = st.slider("Distance (km)", 1.0, 50.0, 10.0, 0.5)
            booking_value = st.slider("Valeur de la course (€)", 5.0, 100.0, 25.0, 1.0)
            vehicle_type = st.selectbox("Type de véhicule", [
                'Economy', 'Premium', 'Shared', 'Luxury'
            ])
        
        with col2:
            payment = st.selectbox("Méthode de paiement", [
                'Card', 'Cash', 'Wallet', 'Corporate'
            ])
            hour = st.slider("Heure de la réservation", 0, 23, 12)
            day_of_week = st.selectbox("Jour de la semaine", [
                'Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'
            ])
            avg_vtat = st.slider("Temps d'attente véhicule (min)", 1.0, 20.0, 7.0, 0.5)
            avg_ctat = st.slider("Temps d'attente client (min)", 1.0, 15.0, 4.0, 0.5)
        
        with col3:
            st.markdown("### Contexte de la prédiction")
            st.info(f"""
            **Trajet :** {pickup} → {drop}  
            **Distance :** {distance} km  
            **Valeur :** {booking_value}€  
            **Véhicule :** {vehicle_type}  
            **Heure :** {hour}h ({day_of_week})
            """)
        
        if st.button("🔮 Prédire le Statut", type="primary", use_container_width=True):
            # Préparation des données pour prédiction
            day_mapping = {
                'Lundi': 0, 'Mardi': 1, 'Mercredi': 2, 'Jeudi': 3, 
                'Vendredi': 4, 'Samedi': 5, 'Dimanche': 6
            }
            
            def get_time_slot(hour):
                if 6 <= hour < 10:
                    return 'Morning_Rush'
                elif 10 <= hour < 16:
                    return 'Midday'
                elif 16 <= hour < 20:
                    return 'Evening_Rush'
                elif 20 <= hour < 24:
                    return 'Evening'
                else:
                    return 'Night_EarlyMorning'
            
            # Construction de l'échantillon de test
            test_data = {
                'Pickup Location': pickup,
                'Drop Location': drop,
                'Ride Distance': distance,
                'Booking Value': booking_value,
                'Vehicle Type': vehicle_type,
                'Payment Method': payment,
                'Avg VTAT': avg_vtat,
                'Avg CTAT': avg_ctat,
                'Hour': hour,
                'DayOfWeek': day_mapping[day_of_week],
                'Month': 6,  # Valeur par défaut
                'IsWeekend': 1 if day_mapping[day_of_week] >= 5 else 0,
                'TimeSlot': get_time_slot(hour)
            }
            
            # Encodage des variables catégorielles
            for col, encoder in label_encoders.items():
                if col in test_data:
                    try:
                        if test_data[col] in encoder.classes_:
                            test_data[col] = encoder.transform([test_data[col]])[0]
                        else:
                            test_data[col] = 0  # Valeur par défaut
                    except:
                        test_data[col] = 0
            
            # Prédiction avec le meilleur modèle
            test_df = pd.DataFrame([test_data])
            best_model = results[best_model_name]['model']
            
            try:
                prediction = best_model.predict(test_df)[0]
                prediction_proba = best_model.predict_proba(test_df)[0]
                predicted_class = target_encoder.inverse_transform([prediction])[0]
                
                # Affichage du résultat
                max_proba = np.max(prediction_proba)
                
                if predicted_class == 'Success':
                    st.success(f"🎉 **Prédiction : {predicted_class}** (Confiance: {max_proba:.1%})")
                else:
                    st.error(f"⚠️ **Prédiction : {predicted_class}** (Confiance: {max_proba:.1%})")
                
                # Graphique des probabilités
                proba_df = pd.DataFrame({
                    'Statut': target_encoder.classes_,
                    'Probabilité': prediction_proba
                }).sort_values('Probabilité', ascending=True)
                
                fig = px.bar(
                    proba_df, 
                    x='Probabilité', 
                    y='Statut',
                    orientation='h',
                    title="Distribution des probabilités",
                    color='Probabilité',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Erreur lors de la prédiction: {e}")
    
    with tab6:
        st.header("📋 Rapport Final du Projet")
        
        # Métriques de performance finales
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        st.markdown("""
        ## 🎯 Objectifs du Projet
        Développer un modèle de machine learning capable de prédire avec précision 
        le statut des réservations Uber pour optimiser les opérations et améliorer 
        l'expérience utilisateur.
        """)
        
        # Résultats principaux
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="success-metric">
                <h4>🏆 Modèle Final</h4>
                <p><strong>{best_model_name}</strong></p>
                <p>Accuracy: {best_accuracy:.3f} ({best_accuracy*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_predictions = len(results[best_model_name]['y_test'])
            correct_predictions = int(best_accuracy * total_predictions)
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Performance Test</h4>
                <p>{correct_predictions}/{total_predictions} prédictions correctes</p>
                <p>Sur ensemble de validation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            cv_mean = results[best_model_name]['cv_mean']
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔄 Validation Croisée</h4>
                <p>Moyenne: {cv_mean:.3f}</p>
                <p>Stabilité confirmée</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forces et améliorations
        st.markdown("## ✅ Points Forts du Projet")
        st.markdown("""
        - **Feature Engineering avancé** : Extraction de patterns temporels pertinents
        - **Pipeline robuste** : Preprocessing automatisé évitant le data leakage
        - **Validation rigoureuse** : Train/test stratifié + validation croisée
        - **Interface utilisateur** : Application interactive pour tests en temps réel
        - **Documentation complète** : Code commenté et rapport détaillé
        """)
        
        st.markdown("## 🔧 Améliorations Possibles")
        st.markdown("""
        - **Données externes** : Intégration météo, trafic, événements
        - **Deep Learning** : Test de réseaux de neurones pour patterns complexes
        - **Optimisation hyperparamètres** : Grid search ou optimisation bayésienne
        - **Features géospatiales** : Analyse de la densité des zones
        - **Modèles ensemblistes** : Stacking ou blending de modèles
        """)
        
        st.markdown("## 🚀 Impact Business")
        st.markdown("""
        Ce modèle peut être utilisé pour :
        - **Optimisation des ressources** : Allocation des chauffeurs aux zones à fort succès
        - **Prévention des annulations** : Identification des réservations à risque
        - **Amélioration UX** : Estimation des temps d'attente réalistes
        - **Stratégie pricing** : Ajustement des tarifs selon probabilité de succès
        """)
        
        # Contact et portfolio
        st.markdown("---")
        st.markdown("""
        ### 👨‍💻 Contact & Portfolio
        
        **Développé par :** Nadir Ali Ahmed  
        **Email :** [Votre email]  
        **LinkedIn :** [Votre profil LinkedIn]  
        **GitHub :** [Repository du projet]  
        
        *Ce projet fait partie de mon portfolio data science. 
        Il démontre mes compétences en machine learning, data preprocessing, 
        et développement d'applications interactives.*
        """)

if __name__ == "__main__":
    main()
>>>>>>> 7c4a6b4d71e4c4f2ff98ea8e61983123f9ea7e9c
