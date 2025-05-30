import logging
import os
import json # Potrzebny do parsowania metadanych JSON
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession 
from livekit.plugins import openai, silero, deepgram, cartesia

# ----- Konfiguracja Logowania -----
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG) 
logger = logging.getLogger("listen-and-respond")
logger.setLevel(logging.DEBUG) 
# ---------------------------------

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
logger.info("Załadowano zmienne środowiskowe z .env, jeśli plik istnieje.")

# --- Twoje ID Głosów Cartesia ---
MALE_VOICE_ID_CARTESIA = "4ef93bb3-682a-46e6-b881-8e157b6b4388"
FEMALE_VOICE_ID_CARTESIA = "575a5d29-1fdc-4d4e-9afa-5a9a71759864"
# Upewnij się, że ten model jest odpowiedni dla obu głosów i języka polskiego
CARTESIA_TTS_MODEL = "sonic-2-2025-05-08" 
# --------------------------------

# Instrukcje dla LLM - pozostają takie same dla obu płci (zgodnie z Twoim życzeniem)
AGENT_INSTRUCTIONS = r"""
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
"""

class SimpleAgent(Agent):
    # Zmieniamy __init__, aby przyjmował tylko konfigurację TTS
    def __init__(self, tts_plugin_config: dict) -> None: # Zmieniłem nazwę argumentu dla jasności
        logger.debug(f"Inicjalizacja SimpleAgent z konfiguracją TTS: {tts_plugin_config}")
        super().__init__(
            instructions=AGENT_INSTRUCTIONS, # Używamy globalnych instrukcji
            stt=deepgram.STT(
                model="nova-2-general",
                language="pl",
                interim_results=True,
                smart_format=True,
                punctuate=True,
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(**tts_plugin_config), # Rozpakowujemy dynamiczną konfigurację TTS
            vad=silero.VAD.load()
        )
        logger.info("SimpleAgent zainicjalizowany pomyślnie.")

    async def on_enter(self):
        session_id_str = str(self.session.id) if self.session and hasattr(self.session, 'id') else "UnknownSession"
        agent_id_str = str(self.id) if hasattr(self, 'id') else "UnknownAgent" # Poprawka: self.id zamiast self.session.agent.id
        logger.info(f"Agent {agent_id_str} on_enter wywołane dla sesji {session_id_str}")
        try:
            self.session.generate_reply()
            logger.info(f"Agent {agent_id_str} wywołał generate_reply() w on_enter dla sesji {session_id_str}")
        except Exception as e:
            logger.error(f"Agent {agent_id_str} - Błąd w on_enter podczas generate_reply(): {e}", exc_info=True)


async def entrypoint(ctx: JobContext):
    job_id = str(ctx.job.id) if hasattr(ctx.job, 'id') else "UnknownJob"
    room_name_str = str(ctx.room.name) if ctx.room and hasattr(ctx.room, 'name') else "UnknownRoom"
    logger.info(f"Job {job_id} otrzymany, łączenie z pokojem {room_name_str}")
    
    # --- Odczytanie wyboru osobowości z metadanych zadania ---
    personality_choice = "female" # Domyślna wartość, jeśli nic nie przyjdzie
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            job_personality = metadata.get("personality")
            if job_personality in ["male", "female"]:
                 personality_choice = job_personality
            logger.info(f"Job {job_id} - Wybrana osobowość z metadanych: {personality_choice}")
        except json.JSONDecodeError as e:
            logger.error(f"Job {job_id} - Błąd parsowania metadanych JSON z metadanych zadania: '{ctx.job.metadata}'. Błąd: {e}. Używam domyślnej osobowości: {personality_choice}")
        except Exception as e: # Ogólniejszy wyjątek, jeśli metadata nie jest stringiem
            logger.error(f"Job {job_id} - Nieoczekiwany typ metadanych zadania lub inny błąd: {e}. Metadane: {ctx.job.metadata}. Używam domyślnej osobowości: {personality_choice}")

    else:
        logger.info(f"Job {job_id} - Brak metadanych zadania. Używam domyślnej osobowości: {personality_choice}")
    # ---------------------------------------------------------

    # --- Dynamiczna konfiguracja TTS na podstawie osobowości ---
    # Model Cartesia i inne parametry pozostają takie same, zmieniamy tylko 'voice' ID
    selected_voice_id = FEMALE_VOICE_ID_CARTESIA
    # Możesz zdefiniować różne emocje/prędkość dla różnych głosów, jeśli chcesz
    # Poniżej zostawiam Twoje oryginalne ustawienia 'emotion' i 'speed'
    # jeśli chcesz je różnicować, musisz dodać logikę if/else podobnie jak dla selected_voice_id
    
    if personality_choice == "male":
        selected_voice_id = MALE_VOICE_ID_CARTESIA
        logger.info(f"Job {job_id} - Konfiguruję agenta jako MĘŻCZYZNA (głos: {selected_voice_id}).")
    else: # Domyślnie lub jeśli 'female'
        logger.info(f"Job {job_id} - Konfiguruję agenta jako KOBIETA (głos: {selected_voice_id}).")

    # Twoja oryginalna konfiguracja Cartesia, tylko 'voice' jest dynamiczny
    tts_plugin_configuration = {
        "model": CARTESIA_TTS_MODEL, 
        "voice": selected_voice_id,    
        "speed": "slow", # Zgodnie z Twoim kodem
        "language": "pl", # Zgodnie z Twoim kodem
        "emotion": ["curiosity:low", "positivity:high", "surprise:high"] # Zgodnie z Twoim kodem
    }
    # ------------------------------------------------------
    
    try:
        await ctx.connect()
        logger.info(f"Job {job_id} pomyślnie połączony z pokojem {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    session = AgentSession() 
    logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony: {session}")

    # Tutaj możesz dodać handlery zdarzeń, które przygotowałem wcześniej,
    # jeśli chcesz mieć bardziej szczegółowe logi z cyklu życia sesji.
    # Na przykład:
    @session.on("llm_response_ended")
    async def on_llm_response_ended(summary: str): # summary to pełna odpowiedź LLM
        logger.info(f"Job {job_id} - Event: LLM Response Ended. Pełna odpowiedź LLM: '{summary}'")

    @session.on("tts_input_prepared")
    async def on_tts_input_prepared(data): # Użyj 'data: TTSInput', jeśli rozwiążesz problem z importem
        text_to_speak = data.text if hasattr(data, 'text') else "Brak tekstu w TTSInput"
        logger.info(f"Job {job_id} - Event: TTS Input Prepared. Tekst dla TTS: '{text_to_speak}'")

    @session.on("agent_speech_started")
    async def on_agent_speech_started():
        logger.info(f"Job {job_id} - Event: Agent Speech Started (agent zaczął mówić)")
    
    # Możesz dodać więcej handlerów, jeśli potrzebujesz

    try:
        logger.info(f"Job {job_id} - Rozpoczynanie AgentSession z dynamicznie skonfigurowanym SimpleAgent...")
        await session.start(
            agent=SimpleAgent(tts_plugin_config=tts_plugin_configuration), 
            room=ctx.room
        )
        logger.info(f"Job {job_id} - AgentSession wystartowała pomyślnie dla pokoju {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} - Krytyczny błąd podczas inicjalizacji lub startu AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} - Kończenie entrypoint.")


if __name__ == "__main__":
    logger.info("Uruchamianie aplikacji CLI dla workera agenta (psycholog-agent)...")
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="psycholog-agent" # Ważne, aby pasowało do dispatch z Vercel
    ))