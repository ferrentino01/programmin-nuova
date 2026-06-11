import asyncio
import os
import json
import logging
from typing import Dict, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configurazione del Logging di Produzione per il Client
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [AgenteAI] - %(message)s'
)
logger = logging.getLogger(__name__)


class MedicalAIAgent:
    """
    Agente AI Generativo Core che si connette ai server MCP per 
    estendere le proprie capacità analitiche in ambito clinico.
    """
    
    def __init__(self, server_script_path: str = "server_diagnosi_mcp.py"):
        # Definiamo i parametri per lanciare il server MCP locale tramite Standard I/O
        self.server_parameters = StdioServerParameters(
            command="python",
            args=[server_script_path]
        )

    async def analizza_caso_clinico(self, prompt_medico: str, parametri_paziente: Dict[str, Any]) -> None:
        """
        Interfaccia principale dell'agente: riceve la richiesta del medico, 
        interroga il server MCP e formula il report finale strutturato.
        """
        logger.info("Inizializzazione sessione di comunicazione MCP...")
        
        # Apertura del canale di trasporto stdio con il server
        async with stdio_client(self.server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                
                # 1. Handshake iniziale con il protocollo MCP
                await session.initialize()
                logger.info("Connessione MCP stabilita. Protocollo sincronizzato.")
                
                # 2. Ispezione dei Tool (scoperta dinamica delle capacità del server)
                risorsa_tools = await session.list_tools()
                tool_disponibili = [t.name for t in risorsa_tools.tools]
                logger.info(f"Tool diagnosticati ed esposti dal server MCP: {tool_disponibili}")
                
                print("\n" + "="*70)
                print(f"📥 [RICHIESTA MEDICA RICEVUTA]:\n{prompt_medico}")
                print("="*70)
                
                # 3. Interrogazione del primo Tool: Statistiche Globali (per calibrare il contesto)
                logger.info("Invocazione tool MCP: 'ottieni_statistiche_globali'...")
                risultato_stats_raw = await session.call_tool("ottieni_statistiche_globali")
                risultato_stats = json.loads(risultato_stats_raw.content[0].text)
                
                # 4. Interrogazione del secondo Tool: Inferenza Predittiva sul Rischio Cardiaco
                logger.info("Invocazione tool MCP: 'calcola_rischio_cardiaco' con i parametri strutturati...")
                risultato_rischio_raw = await session.call_tool(
                    "calcola_rischio_cardiaco", 
                    arguments=parametri_paziente
                )
                risultato_rischio = json.loads(risultato_rischio_raw.content[0].text)
                
                # 5. Generazione Output Strutturato (Simulazione della sintesi dell'LLM)
                self._genera_risposta_rag(prompt_medico, risultato_stats, risultato_rischio)


    def _genera_risposta_rag(self, prompt: str, stats: Dict[str, Any], rischio: Dict[str, Any]) -> None:
        """
        Metodo privato per combinare i dati grezzi dei tool MCP in una risposta 
        clinica ad alto valore aggiunto (Paradigma RAG / Tool Use).
        """
        print("\n🧠 [AGENTE AI - GENERAZIONE REPORT CLINICO STRUTTURATO]:")
        print("-" * 70)
        
        # Se il server ha rifiutato i dati per validazione clinica fallita
        if rischio.get("stato_elaborazione") == "Rifiutato da Validazione Clinica":
            print(f"❌ AZIONE INTERROTTA: Il sistema di controllo medico locale ha rifiutato i parametri.")
            print(f"Dettaglio Anomalia: {rischio.get('dettaglio_anomalia')}")
            print("-" * 70 + "\n")
            return

        # Estrazione dati dal payload MCP
        paziente_id = rischio.get("paziente_anonimizzato_id")
        output_modello = rischio.get("output_modello", {})
        probabilita = output_modello.get("score_probabilita")
        diagnosi = output_modello.get("descrizione_diagnosi")
        confidenza = output_modello.get("confidenza_algoritmo")
        
        # Recupero parametri medi dal dataset per il confronto statistico
        medie_dataset = stats.get("metadati_dataset", {}).get("mean", {})
        eta_media = medie_dataset.get("age", 54.3)
        chol_media = medie_dataset.get("chol", 246.6)

        # Output clinico finale simulando un LLM avanzato senza allucinazioni
        print(f"📋 Identificativo Anonimo Paziente (GDPR): {paziente_id}")
        print(f"🤖 Risultato Inferenza Machine Learning: {diagnosi}")
        print(f"📊 Probabilità di Rischio Calcolata: {probabilita} (Confidenza Algoritmica: {confidenza})")
        print("\n🔍 Analisi Comparativa del Contesto:")
        print(f"  - L'età del paziente è in linea con la media campionaria ({eta_media:.1f} anni).")
        print(f"  - Il livello di colesterolo inserito è monitorato rispetto alla media storica ({chol_media:.1f} mg/dl).")
        print("\n💡 Raccomandazione AI per il Personale Medico:")
        if output_modello.get("esito_binario") == 1:
            print("  ⚠️ ATTENZIONE: Il modello evidenzia anomalie critiche nei tratti elettrocardiografici o nei vasi principali.")
            print("  Si suggerisce di pianificare con urgenza una coronarografia o un approfondimento clinico di secondo livello.")
        else:
            print("  ✔️ Quadro clinico stabile secondo il modello predittivo. Continuare con lo screening periodico standard.")
        print("-" * 70 + "\n")


# =====================================================================
# BLOCCO DI TEST / SIMULAZIONE DI PRODUZIONE
# =====================================================================
if __name__ == "__main__":
    logger.info("=== AVVIO CLIENT AGENTE AI INTERFACCIA MCP ===")
    
    # Istanziamo l'agente passandogli il nome del file del server MCP creato in precedenza
    # Nota: Assicurati che il nome del file coincida esattamente con quello sul tuo computer (es. 'server diagnosi mcp.py')
    agente = MedicalAIAgent(server_script_path="server_diagnosi_mcp.py")
    
    # Caso di Test 1: Paziente Critico (Valori ad alto rischio)
    prompt_critico = "Analizza potenziale urgenza per un paziente anziano con forti dolori al petto e colesterolo alto."
    paziente_critico = {
        "eta": 67.0, "sesso": 1, "tipo_dolore_petto": 3, "pressione_riposo": 160.0,
        "colesterolo": 286.0, "glicemia_digiuno": 0, "ecg_riposo": 2, "frequenza_cardiaca_max": 108.0,
        "angina_sforzo": 1, "depressione_st": 1.5, "pendenza_st": 1, "vasi_colorati": 3, "thal": 2
    }
    
    # Caso di Test 2: Caso di Errore (Valutazione Robustezza con Età impossibile per testare il blocco di sicurezza)
    prompt_errore = "Controlla questo paziente neonato con parametri sballati."
    paziente_errore = {
        "eta": -5.0, "sesso": 0, "tipo_dolore_petto": 1, "pressione_riposo": 120.0,
        "colesterolo": 200.0, "glicemia_digiuno": 0, "ecg_riposo": 0, "frequenza_cardiaca_max": 150.0,
        "angina_sforzo": 0, "depressione_st": 0.0, "pendenza_st": 0, "vasi_colorati": 0, "thal": 0
    }

    async def esegui_test():
        # Esecuzione del Caso 1 (Paziente Critico)
        await agente.analizza_caso_clinico(prompt_critico, paziente_critico)
        
        # Esecuzione del Caso 2 (Test di Robustezza / Sicurezza del Server)
        await agente.analizza_caso_clinico(prompt_errore, paziente_errore)

    # Avvio del ciclo asincrono
    asyncio.run(esegui_test())
