import os
import json
import logging
import hashlib
from typing import Dict, Any
import numpy as np
import joblib
from mcp.server.fastmcp import FastMCP

# Configurazione del Logging di Produzione
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s] - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_PATH = "modello_cardio.pkl"
SCALER_PATH = "scaler_cardio.pkl"
STATS_PATH = "statistiche_dataset.json"

mcp = FastMCP(
    "Advanced-Cardio-Diagnostic-Server",
    version="1.1.0",
    description="Server MCP Enterprise con validazione clinica dei dati e conformità GDPR"
)

_MODEL_CACHE: Any = None
_SCALER_CACHE: Any = None


def _load_ml_resources() -> tuple:
    global _MODEL_CACHE, _SCALER_CACHE
    if _MODEL_CACHE is not None and _SCALER_CACHE is not None:
        return _MODEL_CACHE, _SCALER_CACHE

    logger.info("Caricamento in memoria degli artefatti predittivi (.pkl)...")
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        error_msg = f"Risorse critiche non trovate. Verifica la presenza di '{MODEL_PATH}' e '{SCALER_PATH}'."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        _SCALER_CACHE = joblib.load(SCALER_PATH)
        logger.info("Artefatti ML caricati in cache con successo.")
        return _MODEL_CACHE, _SCALER_CACHE
    except Exception as e:
        logger.critical(f"Fallimento nel caricamento dei file binari serializzati: {str(e)}")
        raise


def _valida_dati_clinici(eta: float, trestbps: float, chol: float, thalach: float) -> None:
    """
    Strato di Validazione Clinica (Business Logic):
    Verifica che i parametri vitali inseriti dall'Agente rientrino in range fisiologici plausibili.
    """
    if not (0 <= eta <= 120):
        raise ValueError(f"Età non plausibile: {eta}. Deve essere compresa tra 0 e 120.")
    if not (40 <= trestbps <= 250):
        raise ValueError(f"Pressione sanguigna a riposo fuori scala clinica: {trestbps} mm Hg.")
    if not (80 <= chol <= 600):
        raise ValueError(f"Livello di colesterolo non plausibile: {chol} mg/dl.")
    if not (30 <= thalach <= 250):
        raise ValueError(f"Frequenza cardiaca massima fuori range: {thalach} bpm.")


def _genera_anonimo_id(eta: float, sesso: int, colesterolo: float) -> str:
    """
    Meccanismo di Anonimizzazione (GDPR Compliance):
    Genera un ID pseudo-anonimo basato su una stringa hash delle caratteristiche fisse del paziente.
    """
    stringa_base = f"paziente_{eta}_{sesso}_{colesterolo}"
    return hashlib.sha256(stringa_base.encode()).hexdigest()[:12].upper()


# =====================================================================
# DEFINIZIONE DEI TOOL MCP
# =====================================================================

@mcp.tool()
def ottieni_statistiche_globali() -> str:
    """
    Restituisce le statistiche descrittive aggregate (medie, deviazioni standard, quantili) 
    del dataset storico di riferimento (Cleveland, Ungheria, Svizzera, Long Beach).
    """
    logger.info("Esecuzione Tool: ottieni_statistiche_globali")
    try:
        if not os.path.exists(STATS_PATH):
            return json.dumps({"stato": "Errore", "messaggio": "File delle statistiche globali non disponibile."}, indent=2)
            
        with open(STATS_PATH, 'r', encoding='utf-8') as f:
            stats_data = json.load(f)
            
        return json.dumps({
            "stato": "Successo",
            "metadati_dataset": stats_data
        }, indent=2)
    except Exception as e:
        logger.error(f"Errore nel recupero delle statistiche: {str(e)}")
        return json.dumps({"stato": "Errore", "dettaglio": str(e)}, indent=2)


@mcp.tool()
def calcola_rischio_cardiaco(
    eta: float, sesso: int, tipo_dolore_petto: int, pressione_riposo: float,
    colesterolo: float, glicemia_digiuno: int, ecg_riposo: int, frequenza_cardiaca_max: float,
    angina_sforzo: int, depressione_st: float, pendenza_st: int, vasi_colorati: int, thal: int
) -> str:
    """
    Esegue un'inferenza predittiva in tempo reale per stimare il rischio di presenza 
    di patologie cardiache nel paziente, basandosi sul modello Random Forest locale.
    Include validazione clinica dei parametri e anonimizzazione dei log.
    """
    # 1. Generazione ID Anonimo per tracciabilità sicura nei log di produzione
    paziente_id = _genera_anonimo_id(eta, sesso, colesterolo)
    logger.info(f"Esecuzione Tool: calcola_rischio_cardiaco per UUID-Paziente: {paziente_id}")
    
    try:
        # 2. Validazione Clinica dei dati in ingresso prima dell'inferenza
        _valida_dati_clinici(eta, pressione_riposo, colesterolo, frequenza_cardiaca_max)
        
        # 3. Caricamento controllato delle risorse ML
        model, scaler = _load_ml_resources()
        
        # 4. Vettorizzazione dell'input
        input_data = np.array([[
            eta, sesso, tipo_dolore_petto, pressione_riposo, colesterolo, glicemia_digiuno,
            ecg_riposo, frequenza_cardiaca_max, angina_sforzo, depressione_st, pendenza_st,
            vasi_colorati, thal
        ]])
        
        # 5. Allineamento dello scaling
        input_scaled = scaler.transform(input_data)
        
        # 6. Inferenza
        predizione_classe = int(model.predict(input_scaled)[0])
        probabilita_rischio = model.predict_proba(input_scaled)[0][1]
        
        # Classificazione qualitativa della confidenza della predizione
        livello_confidenza = "ALTA" if (probabilita_rischio > 0.85 or probabilita_rischio < 0.15) else "MODERATA"
        
        # 7. Payload di risposta arricchito per l'Agente AI
        risultato_diagnostico = {
            "stato_elaborazione": "Successo",
            "paziente_anonimizzato_id": paziente_id,
            "output_modello": {
                "esito_binario": predizione_classe,
                "descrizione_diagnosi": "Rilevata Presenza di Patologia Cardiaca" if predizione_classe == 1 else "Assenza di Evidenze Patologiche Immediate",
                "score_probabilita": f"{probabilita_rischio * 100:.2f}%",
                "confidenza_algoritmo": livello_confidenza
            },
            "compliance_safety": {
                "validazione_medica": "Superata",
                "crittografia_log": "SHA-256 Anonimo"
            }
        }
        
        return json.dumps(risultato_diagnostico, indent=2, ensure_ascii=False)
        
    except ValueError as ve:
        # Intercettazione specifica degli errori di validazione clinica
        logger.warning(f"Validazione fallita per il paziente {paziente_id}: {str(ve)}")
        return json.dumps({
            "stato_elaborazione": "Rifiutato da Validazione Clinica",
            "paziente_anonimizzato_id": paziente_id,
            "dettaglio_anomalia": str(ve)
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Errore critico durante la fase di inferenza sul paziente {paziente_id}: {str(e)}")
        return json.dumps({
            "stato_elaborazione": "Errore Interno Server",
            "dettaglio_anomalia": str(e)
        }, indent=2)


if __name__ == "__main__":
    logger.info("=== AVVIO SERVER MCP ENTERPRISE (V1.1.0) IN MODALITÀ STDIO ===")
    mcp.run(transport='stdio')
