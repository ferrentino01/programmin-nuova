import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Configurazione del Logging Professionale
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class HeartDiseasePipeline:
    """
    Pipeline MLOps Enterprise per l'ingestione, la trasformazione dei dati 
    e l'addestramento del modello predittivo per il rischio cardiaco.
    """
    
    # Elenco tassonomico delle colonne richieste dal dataset originale (14 attributi)
    REQUIRED_COLUMNS = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inizializza la pipeline con una configurazione centralizzata.
        """
        self.config = config or {
            "data_path": "heart.csv",
            "model_output_path": "modello_cardio.pkl",
            "scaler_output_path": "scaler_cardio.pkl",
            "stats_output_path": "statistiche_dataset.json",
            "test_size": 0.2,
            "random_state": 42,
            "model_hyperparameters": {
                "n_estimators": 100,
                "max_depth": 8,
                "class_weight": "balanced" # Gestisce l'eventuale sbilanciamento delle classi
            }
        }
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None

    def ingest_data(self) -> pd.DataFrame:
        """
        Fase 1: Ingestione dei dati con validazione dello schema.
        """
        path = self.config["data_path"]
        logger.info(f"Avvio fase di ingestione dati dal percorso: {path}")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Il dataset critico '{path}' non è stato trovato.")
            
        try:
            df = pd.read_csv(path)
            logger.info(f"Dati caricati. Record grezzi: {df.shape[0]}, Colonne: {df.shape[1]}")
            
            # Data Validation: Verifica conformità dello schema dei dati
            missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Schema del dataset non valido. Colonne mancanti: {missing_cols}")
                
            return df
        except Exception as e:
            logger.error(f"Fallimento critico durante l'ingestione dati: {str(e)}")
            raise

    def transform_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Fase 2: Pipeline di Trasformazione (Data Cleaning, Handling Duplicates/Nulls, Scaling).
        """
        logger.info("Avvio fase di trasformazione, pulizia e scaling dei dati (ETL)")
        
        try:
            # 1. Gestione dei duplicati
            initial_rows = df.shape[0]
            df = df.drop_duplicates()
            deduplicated_rows = df.shape[0]
            if initial_rows != deduplicated_rows:
                logger.warning(f"Rimossi {initial_rows - deduplicated_rows} record duplicati rilevati nel dataset.")

            # 2. Gestione dei valori nulli tramite imputazione robusta (mediana)
            if df.isnull().sum().sum() > 0:
                logger.warning("Rilevati valori nulli latenti. Applicazione dell'imputazione basata su mediana.")
                df = df.fillna(df.median())

            # 3. Separazione Feature / Target
            X = df.drop(columns=['target'])
            y = df['target']

            # 4. Data Splitting con stratificazione (mantiene le proporzioni delle classi)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config["test_size"], 
                random_state=self.config["random_state"],
                stratify=y
            )
            logger.info(f"Data Split completato. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

            # 5. Feature Scaling (Standardizzazione)
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            logger.info("Standardizzazione delle feature numeriche eseguita con successo.")

            # Generazione preventiva delle statistiche per il server MCP
            self._save_dataset_statistics(df)

            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Errore durante la trasformazione dei dati: {str(e)}")
            raise

    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, y_train: pd.Series, y_test: pd.Series) -> float:
        """
        Fase 3: Addestramento del modello e valutazione quantitativa avanzata.
        """
        logger.info("Avvio addestramento del core predittivo (Random Forest)...")
        
        try:
            params = self.config["model_hyperparameters"]
            self.model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                class_weight=params["class_weight"],
                random_state=self.config["random_state"],
                n_jobs=-1 # Utilizza tutti i core della CPU disponibili per l'addestramento parallelo
            )
            
            self.model.fit(X_train, y_train)
            logger.info("Modello addestrato con successo.")

            # Valutazione Quantitativa
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Log metriche ad alto livello (utili per il monitoraggio aziendale)
            logger.info(f"=== METRICHE DI PERFORMANCE ===")
            logger.info(f"Accuratezza Globale: {accuracy * 100:.2f}%")
            
            # Matrice di Confusione per l'analisi dei Falsi Positivi / Falsi Negativi (critici in medicina)
            cm = confusion_matrix(y_test, y_pred)
            logger.info(f"Matrice di Confusione:\nTN: {cm[0][0]} | FP: {cm[0][1]}\nFN: {cm[1][0]} | TP: {cm[1][1]}")
            
            print("\n" + "="*60)
            print("REPORT DI CLASSIFICAZIONE DETTAGLIATO (PRODUZIONE):")
            print("="*60)
            print(classification_report(y_test, y_pred, target_names=["Assenza Patologia (0)", "Presenza Patologia (1)"]))
            print("="*60 + "\n")
            
            return accuracy
        except Exception as e:
            logger.error(f"Errore durante l'addestramento o la valutazione: {str(e)}")
            raise

    def export_artifacts(self) -> None:
        """
        Fase 4: Esportazione sicura degli artefatti binari di persistenza per il server MCP.
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Impossibile esportare: modello o scaler non inizializzati. Esegui prima il training.")
            
        logger.info("Esportazione serializzata degli artefatti in corso...")
        try:
            joblib.dump(self.model, self.config["model_output_path"])
            joblib.dump(self.scaler, self.config["scaler_output_path"])
            logger.info(f"Artefatti salvati correttamente: '{self.config['model_output_path']}' e '{self.config['scaler_output_path']}'")
        except Exception as e:
            logger.error(f"Errore nel salvataggio dei file binari: {str(e)}")
            raise

    def _save_dataset_statistics(self, df: pd.DataFrame) -> None:
        """
        Metodo privato per isolare il salvataggio dei metadati statistici in formato JSON.
        """
        try:
            stats = df.describe().to_dict()
            with open(self.config["stats_output_path"], "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4, ensure_ascii=False)
            logger.info(f"Metadati statistici JSON salvati in: {self.config['stats_output_path']}")
        except Exception as e:
            logger.error(f"Errore nel salvataggio delle statistiche JSON: {str(e)}")


if __name__ == "__main__":
    logger.info("=== AVVIO SISTEMA PIPELINE ENTERPRISE ===")
    
    # Pipeline Execution Flow controllato dal blocco principale
    try:
        pipeline = HeartDiseasePipeline()
        
        raw_data = pipeline.ingest_data()
        X_train, X_test, y_train, y_test = pipeline.transform_data(raw_data)
        
        pipeline.train_and_evaluate(X_train, X_test, y_train, y_test)
        pipeline.export_artifacts()
        
        logger.info("=== PIPELINE COMPLETATA CON SUCCESSO SENZA ANOMALIE ===")
    except Exception as e:
        logger.critical(f"La esecuzione della pipeline è fallita. Errore bloccante: {str(e)}", exc_info=True)
