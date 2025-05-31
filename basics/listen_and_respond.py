import logging
import os
import json # Dodano do potencjalnego odczytu metadanych
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import openai, silero, deepgram, cartesia

# ----- Konfiguracja Logowania na Początku Pliku -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)
logger = logging.getLogger("voice-assistant-agent") # Zmieniono nazwę loggera dla jasności
logger.setLevel(logging.DEBUG)
# ----------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
logger.info("Załadowano zmienne środowiskowe z .env, jeśli plik istnieje.")

# --- Konfiguracje Głosów TTS Cartesia ---
# WAŻNE: Uzupełnij 'YOUR_MALE_VOICE_ID' i ewentualnie 'YOUR_MALE_MODEL_ID'
# rzeczywistymi wartościami od Cartesia dla głosu męskiego.
# Parametry 'speed', 'language', 'emotion' mogą być takie same lub różne.
CARTESIA_TTS_CONFIGS = {
    "female": { # Używamy Twoich dotychczasowych ustawień jako żeńskich
        "model": "sonic-2-2025-05-08",
        "voice_id": "575a5d29-1fdc-4d4e-9afa-5a9a71759864", # Upewnij się, że to ID głosu żeńskiego
        "speed": "slow",
        "language": "pl",
        "emotion": ["curiosity:low", "positivity:high", "surprise:high"]
    },
    "male": {
        "model": "sonic-2-2025-05-08", # Użyj odpowiedniego modelu dla głosu męskiego, jeśli inny
        "voice": "4ef93bb3-682a-46e6-b881-8e157b6b4388", # <<< WAŻNE: ZASTĄP TO ID GŁOSU MĘSKIEGO
        "speed": "slow",
        "language": "pl",
        "emotion": ["curiosity:low", "positivity:high", "surprise:high"] # Dostosuj emocje, jeśli potrzeba
    },
    "default": { # Konfiguracja domyślna, jeśli nie uda się określić głosu
        "model": "sonic-2-2025-05-08",
        "voice": "3d335974-4c4a-400a-84dc-ebf4b73aada6", # Domyślnie np. żeński
        "speed": "slow",
        "language": "pl",
        "emotion": ["curiosity:low", "positivity:high", "surprise:high"]
    }
}
# -----------------------------------------

class SimpleAgent(Agent):
    def __init__(self, tts_plugin: cartesia.TTS) -> None: # Zmieniono: przyjmuje tts_plugin
        logger.debug("Inicjalizacja SimpleAgent...")
        super().__init__(
            instructions=r"""
                Jesteś głosowym asystentem AI, który działa jak wspierający kumpel z zacięciem terapeutycznym, pomagający młodym ludziom (15-30 lat) ogarnąć emocje, problemy społeczne czy życiowe. Rozmawiasz tak, jakbyś siedział/a z kimś na kawie – prosto, naturalnie, bez sztywnych formułek. Używasz codziennego języka, czasem z lekkim slangiem, ale tylko takim, który brzmi autentycznie dla młodych (np. „okej”, „spoko”). Inspirujesz się technikami CBT, ACT i mindfulness, tłumacząc je w prosty sposób, jak dla znajomego. Budujesz zaufanie przez ciepły, empatyczny ton i szybkie przejście do praktycznych rad. Nie zasypujesz pytaniami – słuchasz, odnosisz się do emocji i dajesz tipy, które można od razu zastosować.
Kluczowe Zasady

Ton: Ciepły, luźny, jak kumpel, który chce pomóc. Zero klinicznego języka czy naciąganych tekstów.
Budowanie zaufania: Reagujesz na emocje od razu, np. „Kurczę, to brzmi ciężko. Jak to przeżywasz?” zamiast wielu pytań.
Praktyczne podejście: Dajesz proste, konkretne rady, np. „Jak czujesz, że Cię przytłacza, weź trzy głębokie oddechy i pomyśl o czymś małym, co możesz teraz zrobić.”
Etyka: Jesteś wsparciem, ale nie terapeutą. Przy poważnych tematach (np. myśli samobójcze) mówisz: „To brzmi bardzo poważnie. Może pogadaj z kimś bliskim albo zadzwoń na [infolinia kryzysowa]?”
Język: Codzienny, prosty polski, z poprawną odmianą imion (np. „Aniu”, „Kubo”). Bez imienia? Używasz „Hej, co słychać?”.

Jak Zaczynać
Zacznij od swobodnego powitania: „Hej! Jestem tu, żeby pogadać o tym, co Cię gryzie. Jak masz na imię?” Po imieniu zwracaj się naturalnie, np. „Cześć, Kuba! Co u Ciebie?” Jeśli imienia brak, idź w: „Hej, co się dzieje?”
Przebieg Rozmowy

Start i zaufanie:

Reaguj na emocje, np.: „Słychać, że coś Cię męczy. Chcesz opowiedzieć, co się dzieje?”
Pokazuj, że słuchasz: „Okej, to brzmi jak spora sprawa. Jak się z tym czujesz?”


Ton i dopasowanie:

Lżejsze tematy: Utrzymuj swobodę, np. „To brzmi, jakby życie rzuciło Ci niezłą zagwozdkę. Jak to teraz wygląda?”
Poważne tematy: Empatyczny, ale nie sztywny ton, np. „Kurczę, to musiało Cię mocno przytłoczyć. Jak sobie z tym radzisz?”


Konkretne rady:

Proste tipy od razu, np.:

Oddech: „Jak czujesz, że głowa się gotuje, weź trzy głębokie oddechy – wdech, wydech, to działa jak reset.”
CBT: „Czasem mózg podsuwa czarne scenariusze. Spróbuj pomyśleć: ‘Czy to na pewno prawda, czy tylko mi się wydaje?’”
Mindfulness: „Spróbuj na chwilę się zatrzymać i posłuchać, co się dzieje wokół. To jak wcisnąć pauzę.”


Max 1-2 pytania na turę, np. „Co myślisz, żeby spróbować tego?”


Zakończenie:

Krótko podsumuj: „<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>I</mi><mi>m</mi><mi>i</mi><mtext>ę</mtext></mrow><annotation encoding="application/x-tex">Imię</annotation></semantics></math>, szacun, że się podzieliłeś/aś. Spróbuj tego, o czym mówiliśmy, i daj znać, jak poszło, okej?”
Zachęcaj do powrotu: „Jakby co, jestem tu, pogadamy.”



Etyka i Bezpieczeństwo

Mów, że nie jesteś terapeutą: „Jestem tu, żeby pomóc, ale jak coś jest bardzo poważne, warto pogadać z kimś, kto się na tym zna.”
Przy wzmiankach o samookaleczeniu/myślach samobójczych: „<math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>I</mi><mi>m</mi><mi>i</mi><mtext>ę</mtext></mrow><annotation encoding="application/x-tex">Imię</annotation></semantics></math>, to brzmi naprawdę poważnie. Proszę, pogadaj z kimś zaufanym albo zadzwoń na [np. 116 123]. Jestem z Tobą, ale potrzebujesz wsparcia na żywo.”
Szanuj granice – nie naciskaj, jeśli użytkownik nie chce mówić.
            """,
            stt=deepgram.STT(
                model="nova-2-general",
                language="pl",
                interim_results=True,
                smart_format=True,
                punctuate=True,
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=tts_plugin, # Zmieniono: używa przekazanego pluginu
            vad=silero.VAD.load()
        )
        logger.info(f"SimpleAgent zainicjalizowany pomyślnie z modelem TTS: {tts_plugin.model} i głosem: {tts_plugin.voice}") # Zaktualizowano log

    async def on_enter(self):
        session_id_str = str(self.session.id) if self.session and hasattr(self.session, 'id') else "UnknownSession"
        agent_id_str = str(self.session.agent.id) if self.session and hasattr(self.session, 'agent') and hasattr(self.session.agent, 'id') else "UnknownAgent"
        logger.info(f"Agent {agent_id_str} on_enter wywołane dla sesji {session_id_str}")
        try:
            # Generowanie odpowiedzi na wejściu, używając instrukcji systemowych (promptu agenta)
            # To spowoduje, że agent przywita się zgodnie z "Jak Zaczynać" w instrukcjach.
            await self.session.generate_reply()
            logger.info(f"Agent {agent_id_str} wywołał generate_reply() w on_enter dla sesji {session_id_str}")
        except Exception as e:
            logger.error(f"Agent {agent_id_str} - Błąd w on_enter podczas generate_reply(): {e}", exc_info=True)


async def entrypoint(ctx: JobContext):
    job_id = str(ctx.job.id) if hasattr(ctx.job, 'id') else "UnknownJob"
    room_name_str = str(ctx.room.name) if ctx.room and hasattr(ctx.room, 'name') else "UnknownRoom"
    logger.info(f"Job {job_id} otrzymany, próba połączenia z pokojem: {room_name_str}")

    # --- Ustalanie konfiguracji głosu TTS ---
    selected_voice_key = "default" # Domyślna wartość
    
    # 1. Sprawdź nazwę pokoju
    if room_name_str == "voice-assistant-room-male":
        selected_voice_key = "male"
        logger.info(f"Job {job_id}: Wykryto pokój dla głosu męskiego ({room_name_str}).")
    elif room_name_str == "voice-assistant-room-female":
        selected_voice_key = "female"
        logger.info(f"Job {job_id}: Wykryto pokój dla głosu żeńskiego ({room_name_str}).")
    else:
        logger.warning(f"Job {job_id}: Nieznana nazwa pokoju ({room_name_str}). Próba odczytu metadanych zadania.")
        # 2. (Opcjonalnie) Sprawdź metadane zadania, jeśli nazwa pokoju nie jest decydująca
        # To jest bardziej elastyczne, jeśli chcesz przekazywać 'voice' w inny sposób niż przez nazwę pokoju.
        if ctx.job.metadata:
            try:
                metadata = json.loads(ctx.job.metadata)
                job_voice_preference = metadata.get("voice") # Zakładamy, że frontend wysyła 'voice': 'male'/'female'
                if job_voice_preference in CARTESIA_TTS_CONFIGS:
                    selected_voice_key = job_voice_preference
                    logger.info(f"Job {job_id}: Odczytano preferencję głosu '{selected_voice_key}' z metadanych zadania.")
                elif job_voice_preference:
                    logger.warning(f"Job {job_id}: Nieznana wartość głosu '{job_voice_preference}' w metadanych. Użycie domyślnego.")
            except json.JSONDecodeError:
                logger.warning(f"Job {job_id}: Błąd dekodowania JSON z metadanych zadania: {ctx.job.metadata}")
        else:
            logger.info(f"Job {job_id}: Brak metadanych zadania. Użycie domyślnego głosu '{selected_voice_key}'.")

    tts_config = CARTESIA_TTS_CONFIGS[selected_voice_key]
    logger.info(f"Job {job_id}: Wybrana konfiguracja TTS ({selected_voice_key}): Model={tts_config['model']}, VoiceID={tts_config['voice_id']}")

    # Utworzenie instancji pluginu TTS z wybraną konfiguracją
    try:
        tts_plugin = cartesia.TTS(
            model=tts_config["model"],
            voice=tts_config["voice_id"], # Zmieniono z 'voice' na 'voice_id', aby pasowało do Cartesia i naszego configu
            speed=tts_config["speed"],
            language=tts_config["language"],
            # emotion=tts_config["emotion"] # Odkomentuj, jeśli chcesz używać emocji
        )
        logger.debug(f"Job {job_id}: Plugin Cartesia TTS utworzony pomyślnie dla głosu '{selected_voice_key}'.")
    except Exception as e:
        logger.error(f"Job {job_id}: Nie udało się utworzyć pluginu Cartesia TTS dla głosu '{selected_voice_key}': {e}", exc_info=True)
        logger.error(f"Job {job_id}: Sprawdź, czy masz poprawny CARTESIA_API_KEY w zmiennych środowiskowych.")
        # Jeśli nie można utworzyć TTS, nie ma sensu kontynuować
        return
    # -----------------------------------------

    try:
        await ctx.connect()
        logger.info(f"Job {job_id} pomyślnie połączony z pokojem {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    try:
        session = AgentSession()
        logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony: {session}")

        agent_instance = SimpleAgent(tts_plugin=tts_plugin) # Przekazanie skonfigurowanego TTS

        logger.info(f"Job {job_id} - Rozpoczynanie AgentSession z SimpleAgent (głos: {selected_voice_key})...")
        await session.start(
            agent=agent_instance,
            room=ctx.room
        )
        logger.info(f"Job {job_id} - AgentSession wystartowała pomyślnie dla pokoju {room_name_str} z głosem {selected_voice_key}")

    except Exception as e:
        logger.error(f"Job {job_id} - Krytyczny błąd podczas inicjalizacji lub startu AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} - Kończenie entrypoint. AgentSession (jeśli wystartowała) zakończy pracę, gdy pokój zostanie zamknięty lub agent się rozłączy.")


if __name__ == "__main__":
    logger.info("Uruchamianie aplikacji CLI dla workera agenta...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

