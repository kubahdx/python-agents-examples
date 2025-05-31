import os
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional
import asyncio

from livekit.agents import (
    JobContext,
    JobType,
    WorkerOptions,
    cli,
    Agent,
    AgentSession,
)
from livekit.plugins import cartesia, deepgram, openai, silero # Zaimportuj potrzebne pluginy

# Ustawienie loggera
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("voice-assistant-agent")

# Ładowanie zmiennych środowiskowych
load_dotenv()
logger.info("Załadowano zmienne środowiskowe z .env, jeśli plik istnieje.")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY")
if not CARTESIA_API_KEY:
    logger.warning("CARTESIA_API_KEY nie jest ustawiony w zmiennych środowiskowych. Plugin Cartesia TTS może nie działać poprawnie.")

# Poprawione klucze: 'model' zamiast 'model_id' i 'voice' zamiast 'voice_id'
CARTESIA_TTS_CONFIGS = {
    "male": {
        "model": "sonic-2-2025-05-08",  # Upewnij się, że to poprawny model Cartesia
        "voice": "YOUR_MALE_VOICE_ID_HERE",  # WAŻNE: ZASTĄP TO ID GŁOSU MĘSKIEGO
        "speed": 0.9,
        "language": "pl", # Ustaw odpowiedni język, jeśli model go wspiera
        "emotion": ["positivity:medium"]
    },
    "female": {
        "model": "sonic-2-2025-05-08",  # Upewnij się, że to poprawny model Cartesia
        "voice": "YOUR_FEMALE_VOICE_ID_HERE",  # WAŻNE: ZASTĄP TO ID GŁOSU ŻEŃSKIEGO
        "speed": 1.0,
        "language": "pl", # Ustaw odpowiedni język, jeśli model go wspiera
        "emotion": ["neutral"]
    },
    "default": {
        "model": "sonic-2-2025-05-08",   # Upewnij się, że to poprawny model Cartesia
        "voice": "3d335974-4c4a-400a-84dc-ebf4b73aada6", # Przykładowe ID głosu Cartesia, może wymagać zmiany
        "speed": 1.0,
        "language": "pl", # Ustaw odpowiedni język, jeśli model go wspiera
        "emotion": ["neutral"]
    }
}

class SimpleAgent(Agent):
    def __init__(self, tts_plugin: cartesia.TTS, llm_plugin: openai.LLM, stt_plugin: deepgram.STT):
        super().__init__()
        self.tts_plugin = tts_plugin
        self.llm_plugin = llm_plugin # Dodajemy LLM i STT do agenta
        self.stt_plugin = stt_plugin
        self.chat_history: List[Dict[str, str]] = [
            {"role": "system", "content": "Jesteś pomocnym asystentem głosowym mówiącym po polsku. Odpowiadaj zwięźle."}
        ]
        self.agent_session: Optional[AgentSession] = None

        # Dostęp do atrybutów .model i .voice obiektu tts_plugin
        # Te atrybuty są ustawiane przez plugin Cartesia na podstawie argumentów konstruktora.
        # Używamy getattr dla bezpieczeństwa, na wypadek gdyby atrybuty nie zostały ustawione.
        tts_model_info = getattr(self.tts_plugin, 'model', 'N/A')
        tts_voice_info = getattr(self.tts_plugin, 'voice', 'N/A') # To jest ID głosu lub embedding
        logger.info(f"SimpleAgent zainicjalizowany pomyślnie z modelem TTS: {tts_model_info} i głosem (ID/Embedding): {tts_voice_info}")

    async def _main_processing_loop(self):
        """Główna pętla przetwarzania agenta po nawiązaniu połączenia."""
        if not self.agent_session:
            logger.error("Sesja agenta niedostępna w głównej pętli.")
            return

        # Przywitanie użytkownika po dołączeniu
        try:
            logger.info("Agent próbuje przywitać użytkownika.")
            # Używamy historii czatu, aby LLM mógł wygenerować odpowiednie powitanie
            # Można też przekazać bezpośredni tekst do `say`
            await self.agent_session.say("Cześć! Jestem Twoim asystentem. Jak mogę pomóc?", allow_interruptions=True)
            logger.info("Wysłano powitanie.")
        except Exception as e:
            logger.error(f"Błąd podczas wysyłania powitania: {e}")

        # AgentSession zajmie się główną pętlą STT->LLM->TTS.
        # Ta pętla może służyć do dodatkowej logiki lub monitorowania.
        while True:
            await asyncio.sleep(1) # Utrzymuj pętlę przy życiu
            if self.agent_session.is_closed: # Sprawdź czy sesja jest zamknięta
                logger.info("Sesja agenta została zamknięta. Zakończenie głównej pętli.")
                break


    async def start(self, session: AgentSession): # Ta metoda jest wywoływana przez AgentSession
        self.agent_session = session
        logger.info(f"SimpleAgent wszedł do sesji dla pokoju: {session.room.name}")
        # Uruchom główną pętlę przetwarzania jako zadanie w tle
        asyncio.create_task(self._main_processing_loop())


async def entrypoint(ctx: JobContext):
    job_id = ctx.job_id
    logger.info(f"Job {job_id} otrzymany, próba połączenia z pokojem: {ctx.room.name}")

    tts_config_choice = "default"
    room_name = ctx.room.name

    if room_name == "voice-assistant-room-male":
        tts_config_choice = "male"
    elif room_name == "voice-assistant-room-female":
        tts_config_choice = "female"
    else:
        logger.warning(f"Job {job_id}: Nieznana nazwa pokoju ({room_name}). Próba odczytu metadanych zadania.")
        if ctx.job.metadata:
            try:
                metadata = json.loads(ctx.job.metadata)
                voice_preference = metadata.get("voice")
                if voice_preference in CARTESIA_TTS_CONFIGS:
                    tts_config_choice = voice_preference
                    logger.info(f"Job {job_id}: Użycie głosu '{tts_config_choice}' z metadanych zadania.")
                else:
                    logger.warning(f"Job {job_id}: Nieznana wartość głosu ('{voice_preference}') w metadanych. Użycie domyślnego głosu.")
            except json.JSONDecodeError:
                logger.error(f"Job {job_id}: Błąd parsowania metadanych JSON. Użycie domyślnego głosu.")
        else:
            logger.info(f"Job {job_id}: Brak metadanych zadania. Użycie domyślnego głosu '{tts_config_choice}'.")

    selected_tts_config = CARTESIA_TTS_CONFIGS[tts_config_choice].copy() # Użyj kopii, aby uniknąć modyfikacji oryginału

    # Sprawdzenie API Key PRZED próbą utworzenia pluginu
    if not CARTESIA_API_KEY:
        logger.error(f"Job {job_id}: Klucz CARTESIA_API_KEY nie jest ustawiony. Nie można utworzyć pluginu Cartesia TTS.")
        return

    # Sprawdzenie placeholderów w wybranej konfiguracji głosu
    current_voice_id = selected_tts_config.get("voice", "")
    if "YOUR_MALE_VOICE_ID_HERE" in current_voice_id or \
       "YOUR_FEMALE_VOICE_ID_HERE" in current_voice_id:
        logger.error(f"Job {job_id}: Placeholder Voice ID ('{current_voice_id}') znaleziony dla konfiguracji '{tts_config_choice}'. "
                       "Proszę zastąpić go rzeczywistym ID głosu w słowniku CARTESIA_TTS_CONFIGS.")
        # Spróbuj użyć domyślnego, jeśli wybrany miał placeholder
        default_voice_id = CARTESIA_TTS_CONFIGS["default"].get("voice", "")
        if "YOUR_" not in default_voice_id:
            logger.warning(f"Job {job_id}: Przełączanie na domyślną konfigurację głosu z powodu placeholdera w '{tts_config_choice}'.")
            selected_tts_config = CARTESIA_TTS_CONFIGS["default"].copy()
            tts_config_choice = "default"
        else:
            logger.error(f"Job {job_id}: Domyślna konfiguracja głosu ('{default_voice_id}') również zawiera placeholder. "
                           "Nie można kontynuować. Proszę ustawić poprawne ID głosów.")
            return

    # Logowanie wybranej konfiguracji (po ewentualnym fallbacku)
    logger.info(f"Job {job_id}: Wybrana konfiguracja TTS ({tts_config_choice}): Model={selected_tts_config.get('model')}, VoiceID={selected_tts_config.get('voice')}")

    try:
        # Plugin Cartesia jest tworzony z rozpakowaną konfiguracją.
        # Konstruktor cartesia.TTS oczekuje 'model', 'voice', 'speed', etc. jako argumenty kluczowe.
        tts_plugin = cartesia.TTS(**selected_tts_config)
        logger.debug(f"Job {job_id}: Plugin Cartesia TTS utworzony pomyślnie dla głosu '{tts_config_choice}'.")
    except Exception as e:
        logger.error(f"Job {job_id}: Krytyczny błąd podczas tworzenia pluginu Cartesia TTS: {e}")
        logger.error(f"Job {job_id}: Użyta konfiguracja: {selected_tts_config}")
        return # Zakończ, jeśli plugin TTS nie może być utworzony

    # Inicjalizacja pozostałych pluginów (STT, LLM) - upewnij się, że masz odpowiednie klucze API
    try:
        # Przykładowe konfiguracje - dostosuj do swoich potrzeb
        if not os.getenv("DEEPGRAM_API_KEY"): # Sprawdź czy klucz jest dostępny
             logger.warning("DEEPGRAM_API_KEY nie jest ustawiony. Plugin Deepgram STT może nie działać.")
        stt_plugin = deepgram.STT(model="nova-2", language="pl", interim_results=True, endpointing_ms=300)

        if not os.getenv("OPENAI_API_KEY"): # Sprawdź czy klucz jest dostępny
             logger.warning("OPENAI_API_KEY nie jest ustawiony. Plugin OpenAI LLM może nie działać.")
        llm_plugin = openai.LLM(model="gpt-4o-mini") # Użyj modelu obsługującego język polski
    except Exception as e:
        logger.error(f"Job {job_id}: Błąd podczas inicjalizacji pluginu STT lub LLM: {e}")
        return

    try:
        await ctx.connect() # Łączenie z pokojem LiveKit
        logger.info(f"Job {job_id} pomyślnie połączony z pokojem {ctx.room.name}")

        # Utworzenie instancji agenta z skonfigurowanymi pluginami
        agent_instance = SimpleAgent(tts_plugin=tts_plugin, llm_plugin=llm_plugin, stt_plugin=stt_plugin)

        # AgentSession zarządza pętlą STT->LLM->TTS.
        # Przekazujemy do niej agenta oraz instancje pluginów.
        session = AgentSession(
            room=ctx.room,
            agent=agent_instance, # Nasza instancja agenta
            stt=stt_plugin,       # Skonfigurowany plugin STT
            llm=llm_plugin,       # Skonfigurowany plugin LLM
            tts=tts_plugin,       # Skonfigurowany plugin TTS
            vad=silero.VAD.load(),# Przykładowy VAD, upewnij się, że masz modele Silero
            # Możesz dodać inne opcje i pluginy tutaj, jeśli są potrzebne
        )
        
        logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony.")
        
        # Uruchomienie sesji agenta. To wywoła metodę `start` w `SimpleAgent`.
        await session.start() 
        logger.info(f"Job {job_id} - AgentSession wystartowała. Agent jest aktywny i nasłuchuje.")

    except Exception as e:
        logger.error(f"Job {job_id} - Krytyczny błąd podczas inicjalizacji lub startu AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} - Kończenie entrypoint. AgentSession (jeśli wystartowała) zarządza swoim cyklem życia.")
        # AgentSession i JobContext powinny zarządzać cyklem życia połączenia.

if __name__ == "__main__":
    logger.info("Sprawdzanie konfiguracji LiveKit...")
    if not os.getenv("LIVEKIT_API_KEY") or \
       not os.getenv("LIVEKIT_API_SECRET") or \
       not os.getenv("LIVEKIT_URL"):
        logger.error("Zmienne środowiskowe LiveKit (LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL) nie są w pełni ustawione. Prerywam.")
    else:
        logger.info("Uruchamianie aplikacji CLI dla workera agenta...")
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, job_type=JobType.JT_ROOM))
