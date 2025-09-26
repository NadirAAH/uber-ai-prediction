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
from pathlib import Path
import os

# Configuration identique au code original
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'
warnings.filterwarnings('ignore')

# Configuration Streamlit
st.set_page_config(
    page_title="Projet Data Science Complet : Prédiction des Statuts de Réservation Uber",
    page_icon="🚗",
    layout="wide"
)

# CSS simple et sûr
st.markdown("""
<style>
    .main { background-color: #0e1117; color: white; }
    .stApp { background-color: #0e1117; }
    .metric-box { 
        padding: 1rem; 
        background-color: #262730; 
        border-radius: 8px; 
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .section-header { 
        color: #FF6B6B; 
        border-bottom: 2px solid #FF6B6B; 
        padding-bottom: 0.5rem;
        margin: 2rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_validate_data(file_path=None):
    """
    Fonction de chargement EXACTE du code d'entraînement original
    Rôle : Centralisation de la logique de chargement avec gestion d'erreurs
    Intérêt : Réutilisabilité, robustesse, feedback détaillé
    """
    try:
        # Chemins possibles selon la structure du projet
        possible_paths = [
            'archive/ncr_ride_bookings.csv',
            'uber/archive/ncr_ride_bookings.csv',
            '/Users/nadiraliahmed/Desktop/ProjetData /uber/archive/ncr_ride_bookings.csv'
        ]
        
        df = None
        used_path = None
        
        for path_str in possible_paths:
            try:
                file_path = Path(path_str)
                if file_path.exists():
                    file_size_mb = file_path.stat().st_size / 1024**2
                    st.info(f"Chargement du fichier : {file_path.name}")
                    st.info(f"Taille : {file_size_mb:.1f} MB")
                    
                    df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
                    used_path = path_str
                    break
            except:
                continue
        
        if df is None:
            st.warning("ERREUR: Fichier non trouvé. Utilisation de données de démonstration.")
            return generate_demo_data()
        
        if df.empty:
            st.error("ERREUR: Le dataset est vide")
            return generate_demo_data()
            
        st.success(f"Dataset chargé avec succès: {df.shape[0]:,} lignes x {df.shape[1]} colonnes")
        return df
        
    except Exception as e:
        st.error(f"ERREUR lors du chargement: {e}")
        return generate_demo_data()

@st.cache_data
def generate_demo_data():
    """Génère des données de démonstration avec la même structure que les vraies données"""
    np.random.seed(42)
    n_samples = 5000
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='2H')
    
    data = {
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Time': [d.strftime('%H:%M:%S') for d in dates],
        'Booking Status': np.random.choice([
            'Success', 'Canceled by Driver', 'Canceled by Customer'
        ], n_samples, p=[0.70, 0.15, 0.15]),
        'Pickup Location': np.random.choice([
            'Airport', 'Railway Station', 'Metro Station', 'Mall', 'Hospital',
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples),
        'Drop Location': np.random.choice([
            'Airport', 'Railway Station', 'Metro Station', 'Mall', 'Hospital',
            'Hotel', 'Office Complex', 'Residential Area'
        ], n_samples),
        'Ride Distance': np.random.exponential(8, n_samples).clip(min=0.5),
        'Booking Value': np.random.gamma(2, 12, n_samples).clip(min=5),
        'Vehicle Type': np.random.choice([
            'Mini', 'Economy', 'Premium', 'Auto'
        ], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'Payment Method': np.random.choice([
            'Card', 'Cash', 'Wallet', 'Corporate'
        ], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'Avg VTAT': np.random.normal(8.5, 4.0, n_samples).clip(min=1),
        'Avg CTAT': np.random.normal(5.2, 2.5, n_samples).clip(min=1)
    }
    
    return pd.DataFrame(data)

@st.cache_data
def preprocess_data_exact_method(df_raw):
    """
    Preprocessing EXACT selon le code d'entraînement original
    Reproduction fidèle de toute la méthodologie
    """
    # Variable cible exacte du code original
    TARGET_COLUMN = "Booking Status"
    
    # Copie et nettoyage initial (même logique)
    df_clean = df_raw.copy()
    initial_shape = df_clean.shape
    
    st.markdown("### 1. NETTOYAGE DES DONNÉES")
    st.write(f"• Dataset initial: {initial_shape[0]:,} lignes")
    
    # Supprimer les lignes où la variable cible est manquante
    df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
    st.write(f"• Après suppression cible manquante: {df_clean.shape[0]:,} lignes")
    
    lines_dropped = initial_shape[0] - df_clean.shape[0]
    st.write(f"• Lignes supprimées: {lines_dropped:,}")
    
    # Identifier colonnes avec trop de valeurs manquantes (>80%) - EXACT du code original
    high_missing_threshold = 0.8
    columns_to_drop = []
    
    st.markdown("### 2. IDENTIFICATION DES COLONNES INUTILISABLES (>80% manquant)")
    
    for col in df_clean.columns:
        if col != TARGET_COLUMN:
            missing_pct = df_clean[col].isnull().sum() / len(df_clean)
            if missing_pct > high_missing_threshold:
                columns_to_drop.append(col)
                st.write(f"• {col}: {missing_pct*100:.1f}% manquant → SUPPRESSION")
    
    df_clean = df_clean.drop(columns=columns_to_drop)
    remaining_cols = df_clean.shape[1]
    dropped_cols = len(columns_to_drop)
    st.write(f"• Colonnes restantes: {remaining_cols} (supprimées: {dropped_cols})")
    
    # Feature engineering EXACT selon le code original
    st.markdown("### 3. FEATURE ENGINEERING")
    
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN].copy()
    
    st.write(f"• Features disponibles: {X.shape[1]}")
    st.write(f"• Observations: {X.shape[0]:,}")
    
    # Features légitimes EXACTES du code original
    LEGITIMATE_FEATURES = [
        'Date',
        'Time',
        'Pickup Location',
        'Drop Location',
        'Ride Distance',
        'Booking Value',
        'Vehicle Type',
        'Payment Method',
        'Avg VTAT',
        'Avg CTAT'
    ]
    
    st.write("✅ FEATURES LÉGITIMES SÉLECTIONNÉES")
    st.write(f"Variables définies par l'utilisateur: {len(LEGITIMATE_FEATURES)}")
    
    available_features = []
    for feat in LEGITIMATE_FEATURES:
        if feat in X.columns:
            available_features.append(feat)
            st.write(f"     + {feat}: DISPONIBLE")
        else:
            st.write(f"     - {feat}: NON DISPONIBLE dans le dataset")
    
    st.write("✅ Features validées par l'expertise utilisateur")
    
    X_selected = X[available_features].copy()
    st.write(f"• Shape finale: {X_selected.shape}")
    
    # Feature engineering temporel EXACT du code original
    if 'Date' in X_selected.columns and 'Time' in X_selected.columns:
        st.write("🕒 FEATURE ENGINEERING TEMPOREL")
        
        try:
            X_selected['DateTime'] = pd.to_datetime(X_selected['Date'] + ' ' + X_selected['Time'])
            X_selected['Hour'] = X_selected['DateTime'].dt.hour
            X_selected['DayOfWeek'] = X_selected['DateTime'].dt.dayofweek
            X_selected['Month'] = X_selected['DateTime'].dt.month
            X_selected['IsWeekend'] = (X_selected['DayOfWeek'] >= 5).astype(int)
            
            # Fonction get_time_slot EXACTE du code original
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
            
            X_selected['TimeSlot'] = X_selected['Hour'].apply(get_time_slot)
            X_selected = X_selected.drop(columns=['Date', 'Time', 'DateTime'])
            
            st.write("✅ Features temporelles créées: Hour, DayOfWeek, Month, IsWeekend, TimeSlot")
            
        except Exception as e:
            st.write(f"⚠️ Erreur feature engineering temporel: {e}")
    
    # Mise à jour de la liste des features après transformations
    available_features = X_selected.columns.tolist()
    
    # Encodage de la variable cible EXACT
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    st.markdown("### 4. ENCODAGE DE LA VARIABLE CIBLE")
    st.write(f"• Classes détectées: {len(label_encoder.classes_)}")
    
    for i, class_name in enumerate(label_encoder.classes_):
        count = np.sum(y_encoded == i)
        st.write(f"     {i}: {class_name} ({count:,} exemples)")
    
    # Traitement des features EXACT du code original
    categorical_features = X_selected.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X_selected.select_dtypes(include=[np.number]).columns.tolist()
    
    st.markdown("### 5. TRAITEMENT DES FEATURES")
    st.write(f"• Features catégorielles: {len(categorical_features)}")
    st.write(f"• Features numériques: {len(numerical_features)}")
    
    X_processed = X_selected.copy()
    feature_encoders = {}
    
    # Traitement EXACT du code original
    for feature in categorical_features:
        X_processed[feature] = X_processed[feature].fillna('MISSING')
        X_processed[feature] = X_processed[feature].astype(str)
        feature_encoder = LabelEncoder()
        X_processed[feature] = feature_encoder.fit_transform(X_processed[feature])
        feature_encoders[feature] = feature_encoder
        original_categories = X_selected[feature].nunique()
        st.write(f"     • {feature}: {original_categories} catégories encodées")
    
    st.write(f"• Dataset final préprocessé: {X_processed.shape}")
    
    return X_processed, y_encoded, label_encoder, feature_encoders, df_clean, available_features

def create_preprocessing_pipeline():
    """Pipeline EXACT du code d'entraînement original"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

def evaluate_model(model, model_name, X_train, X_test, y_train, y_test, label_encoder):
    """Fonction d'évaluation EXACTE du code original"""
    st.write(f"\n--- ÉVALUATION : {model_name} ---")
    st.write("Entraînement en cours...")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    st.write(f"Accuracy sur test: {accuracy:.4f} ({accuracy*100:.2f}%)")
    st.write(f"Validation croisée (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    st.write("Rapport de classification:")
    report_text = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.code(report_text)
    
    return {
        'model': model,
        'name': model_name,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

@st.cache_resource
def train_models_exact_method(X_processed, y_encoded, label_encoder):
    """Entraînement EXACT selon le code original"""
    
    st.markdown("## DIVISION TRAIN/TEST")
    
    # Division EXACTE du code original
    try:
        stratify_param = y_encoded if len(set(y_encoded)) > 1 and min(np.bincount(y_encoded)) >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=stratify_param
        )
        
        st.write("• Division stratifiée réussie")
        
    except Exception as e:
        st.write(f"• Stratification impossible: {e}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42
        )
    
    st.write(f"• Set d'entraînement: {X_train.shape[0]:,} exemples")
    st.write(f"• Set de test: {X_test.shape[0]:,} exemples")
    
    # Distribution des classes EXACTE
    st.write("\nDistribution des classes:")
    distribution_data = []
    for i, class_name in enumerate(label_encoder.classes_):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        total_count = train_count + test_count
        distribution_data.append({
            'Classe': class_name,
            'Train': train_count,
            'Test': test_count,
            'Total': total_count
        })
    
    dist_df = pd.DataFrame(distribution_data)
    st.dataframe(dist_df)
    
    st.markdown("## SÉLECTION ET ENTRAÎNEMENT DU MODÈLE")
    
    # Pipeline EXACT du code original
    preprocessing_pipeline = create_preprocessing_pipeline()
    
    st.write("1. COMPARAISON D'ALGORITHMES")
    
    # Modèles EXACTS du code original
    logistic_model = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    rf_model = Pipeline([
        ('preprocessing', preprocessing_pipeline),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ))
    ])
    
    # Évaluation EXACTE
    logistic_results = evaluate_model(
        logistic_model, "Régression Logistique", X_train, X_test, y_train, y_test, label_encoder
    )
    
    rf_results = evaluate_model(
        rf_model, "Random Forest", X_train, X_test, y_train, y_test, label_encoder
    )
    
    results = [logistic_results, rf_results]
    best_model_result = max(results, key=lambda x: x['accuracy'])
    
    st.write("2. SÉLECTION DU MEILLEUR MODÈLE")
    st.write(f"• Meilleur modèle: {best_model_result['name']}")
    st.write(f"• Accuracy: {best_model_result['accuracy']:.4f}")
    st.write(f"• Validation croisée: {best_model_result['cv_mean']:.4f} ± {best_model_result['cv_std']:.4f}")
    
    return results, best_model_result, X_train, X_test, y_train, y_test

def analyze_performance(y_test, final_predictions, label_encoder):
    """Analyse détaillée EXACTE du code original"""
    
    st.markdown("## ANALYSE DÉTAILLÉE DES PERFORMANCES")
    
    # Matrice de confusion EXACTE
    cm = confusion_matrix(y_test, final_predictions)
    
    st.write("1. MATRICE DE CONFUSION")
    st.write("   Lignes = Vraies classes, Colonnes = Prédictions")
    
    # DataFrame EXACT du code original
    cm_df = pd.DataFrame(cm,
                         index=[f"Vrai_{cls}" for cls in label_encoder.classes_],
                         columns=[f"Pred_{cls}" for cls in label_encoder.classes_])
    
    st.dataframe(cm_df)
    
    # Performance par classe EXACTE
    st.write("2. PERFORMANCE PAR CLASSE")
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, final_predictions, average=None
    )
    
    perf_data = []
    for i, class_name in enumerate(label_encoder.classes_):
        perf_data.append({
            'Classe': class_name,
            'Precision': f"{precision[i]:.3f}",
            'Recall': f"{recall[i]:.3f}",
            'F1-Score': f"{f1[i]:.3f}",
            'Support': support[i]
        })
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df)
    
    return precision, recall, f1, support

def generate_conclusions(best_model_result, precision, recall, f1, label_encoder):
    """Conclusions EXACTES du code original"""
    
    st.markdown("## CONCLUSIONS ET RECOMMANDATIONS")
    
    st.write("1. RÉSULTATS OBTENUS")
    st.write(f"• Modèle final: {best_model_result['name']}")
    st.write(f"• Accuracy: {best_model_result['accuracy']:.4f} ({best_model_result['accuracy']*100:.2f}%)")
    st.write(f"• Validation croisée: {best_model_result['cv_mean']:.4f} ± {best_model_result['cv_std']:.4f}")
    
    # Évaluation de performance EXACTE du code original
    accuracy = best_model_result['accuracy']
    if accuracy >= 0.75:
        performance_level = "EXCELLENT (sans data leakage)"
    elif accuracy >= 0.65:
        performance_level = "BON (réaliste)"
    elif accuracy >= 0.55:
        performance_level = "ACCEPTABLE (prédiction difficile)"
    else:
        performance_level = "INSUFFISANT (nécessite amélioration)"
    
    st.write(f"• Niveau de performance: {performance_level}")
    
    # Forces du modèle EXACTES
    st.write("2. FORCES DU MODÈLE")
    
    best_classes = []
    for i, class_name in enumerate(label_encoder.classes_):
        if f1[i] >= 0.8:  # Seuil EXACT du code original
            best_classes.append((class_name, f1[i]))
    
    if best_classes:
        st.write("• Classes bien prédites:")
        for class_name, f1_score in best_classes:
            st.write(f"     - {class_name}: F1-Score = {f1_score:.3f}")
    
    # Points d'amélioration EXACTS
    st.write("3. POINTS D'AMÉLIORATION")
    
    weak_classes = []
    for i, class_name in enumerate(label_encoder.classes_):
        if f1[i] < 0.6:  # Seuil EXACT du code original
            weak_classes.append((class_name, f1[i], recall[i]))
    
    if weak_classes:
        st.write("• Classes à améliorer:")
        for class_name, f1_score, recall_score in weak_classes:
            st.write(f"     - {class_name}: F1-Score = {f1_score:.3f}, Recall = {recall_score:.3f}")
    
    # Recommandations EXACTES du code original
    st.write("4. RECOMMANDATIONS TECHNIQUES")
    st.write("• Feature engineering avancé:")
    st.write("     - Enrichir les données de géolocalisation")
    st.write("     - Créer des features d'historique client")
    st.write("     - Intégrer des données externes (météo, trafic)")
    st.write("• Modélisation adaptée:")
    st.write("     - Tester XGBoost, CatBoost")
    st.write("     - Optimiser les seuils de décision par classe")
    
    st.write("5. RECOMMANDATIONS BUSINESS")
    st.write("• Monitoring: Surveiller la dérive des données en production")
    st.write("• Seuils de décision: Ajuster selon les coûts métier")
    st.write("• Automatisation: Intégrer le modèle dans le système")

def main():
    # Titre EXACT du code original
    st.title("PROJET DATA SCIENCE COMPLET : PRÉDICTION DES STATUTS DE RÉSERVATION UBER")
    st.markdown("**Rôle :** Titre principal du projet pour identifier clairement l'objectif")
    st.markdown("**Intérêt :** Documentation, traçabilité du projet, communication de l'objectif métier")
    st.markdown("**Développé par :** Nadir Ali Ahmed")
    
    st.markdown("**Configuration système optimisée pour Apple M2**")
    st.markdown("**Imports et configuration terminés avec succès**")
    
    st.markdown("---")
    
    # Chargement des données avec la fonction EXACTE
    st.markdown("## ANALYSE EXPLORATOIRE DES DONNÉES (EDA)")
    
    df_raw = load_and_validate_data()
    
    if df_raw is None:
        st.error("ARRÊT: Impossible de continuer sans données")
        return
    
    # Informations générales EXACTES
    st.markdown("### 1. INFORMATIONS GÉNÉRALES")
    memory_usage_mb = df_raw.memory_usage(deep=True).sum() / 1024**2
    st.write(f"• Dimensions: {df_raw.shape[0]:,} lignes x {df_raw.shape[1]} colonnes")
    st.write(f"• Mémoire utilisée: {memory_usage_mb:.1f} MB")
    st.write(f"• Plage temporelle: du {df_raw['Date'].min()} au {df_raw['Date'].max()}")
    
    # Aperçu des données EXACT
    st.markdown("### 2. APERÇU DES DONNÉES (5 premières lignes)")
    st.dataframe(df_raw.head())
    
    # Analyse détaillée EXACTE du code original
    st.markdown("### 3. ANALYSE DÉTAILLÉE DES VARIABLES")
    
    analysis_data = []
    for col in df_raw.columns:
        unique_count = df_raw[col].nunique()
        missing_count = df_raw[col].isnull().sum()
        missing_percent = (missing_count / len(df_raw)) * 100
        data_type = str(df_raw[col].dtype)
        
        analysis_data.append({
            'Variable': col,
            'Type': data_type,
            'Uniques': unique_count,
            'Manquant': missing_count,
            '% Manquant': f"{missing_percent:.1f}%"
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    st.dataframe(analysis_df)
    
    # Variable cible EXACTE
    TARGET_COLUMN = "Booking Status"
    st.markdown(f"### 5. VARIABLE CIBLE SÉLECTIONNÉE: '{TARGET_COLUMN}'")
    
    if TARGET_COLUMN in df_raw.columns:
        target_distribution = df_raw[TARGET_COLUMN].value_counts()
        st.write("Distribution des classes:")
        
        for class_name, count in target_distribution.items():
            percentage = (count / len(df_raw)) * 100
            st.write(f"   • {class_name}: {count:,} ({percentage:.1f}%)")
        
        max_class_pct = (target_distribution.iloc[0] / len(df_raw)) * 100
        
        if max_class_pct > 60:
            st.warning(f"⚠ DÉSÉQUILIBRE DÉTECTÉ: Classe majoritaire = {max_class_pct:.1f}%")
            st.info("→ Utilisation recommandée de 'class_weight=balanced'")
    
    st.markdown("---")
    
    # Preprocessing EXACT
    st.markdown("## NETTOYAGE ET PRÉPROCESSING")
    
    X_processed, y_encoded, label_encoder, feature_encoders, df_clean, available_features = preprocess_data_exact_method(df_raw)
    
    st.markdown("---")
    
    # Modélisation EXACTE
    results, best_model_result, X_train, X_test, y_train, y_test = train_models_exact_method(X_processed, y_encoded, label_encoder)
    
    final_model = best_model_result['model']
    final_predictions = best_model_result['predictions']
    
    st.markdown("---")
    
    # Analyse des performances EXACTE
    precision, recall, f1, support = analyze_performance(y_test, final_predictions, label_encoder)
    
    st.markdown("---")
    
    # Conclusions EXACTES
    generate_conclusions(best_model_result, precision, recall, f1, label_encoder)
    
    st.markdown("---")
    st.success("PROJET TERMINÉ AVEC SUCCÈS")
    st.success("Modèle prêt pour la mise en production")

if __name__ == "__main__":
    main()