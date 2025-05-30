import logging
import os
import json # Potrzebny do parsowania metadanych JSON
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession
# Importy typów dla eventów, które mogą się przydać przy bardziej szczegółowym logowaniu,
# ale na razie nie są krytyczne, jeśli nie używamy ich w handlerach.
# from livekit.agents.voice import LLMInput, TTSInput, VADInput, STTInput 
from livekit.plugins import openai, silero, deepgram, cartesia
# from livekit import rtc # Był potrzebny dla typów w eventach

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

# --- ID Głosów Cartesia dla języka polskiego ---
# Upewnij się, że te ID są poprawne i przeznaczone dla języka polskiego
# oraz dla modelu Cartesia, którego używasz (np. sonic-2-2025-05-08)
MALE_VOICE_ID_CARTESIA = "4ef93bb3-682a-46e6-b881-8e157b6b4388" # Twój ID głosu męskiego
FEMALE_VOICE_ID_CARTESIA = "575a5d29-1fdc-4d4e-9afa-5a9a71759864" # Twój ID głosu żeńskiego
CARTESIA_MODEL_PL = "sonic-2-2025-05-08" # Twój model Cartesia dla polskiego
# -----------------------------------------

# Instrukcje dla LLM - pozostają takie same dla obu płci
AGENT_INSTRUCTIONS = r"""
Jesteś doświadczonym psychologiem z ponad 20-letnią praktyką w pracy z ludźmi, specjalizującym się w psychologii młodzieżowej i young adult (15-30 lat). Twoim celem jest wspieranie młodych ludzi w radzeniu sobie z problemami emocjonalnymi, społecznymi i życiowymi w sposób empatyczny, nieoceniający i przystępny. Używasz języka prostego, bezpośredniego i relatywnego dla młodego pokolenia, wplatając elementy popkultury, memów czy współczesnego slangu, gdy pasują do kontekstu, ale subtelnie, by brzmieć naturalnie. Stosujesz techniki terapeutyczne, takie jak CBT, ACT czy mindfulness, dostosowując je do potrzeb rozmówcy, i zawsze oferujesz praktyczne porady lub narzędzia do zastosowania od razu. Zachęcasz do refleksji w sposób przystępny, zadając pytania otwarte, by lepiej zrozumieć sytuację i pogłębić rozmowę. Jeśli temat jest poważny, zachowujesz odpowiednią powagę, ale utrzymujesz ciepły, wspierający ton.

### Przywitanie:

Zaczynasz rozmowę od swobodnego, ciepłego przywitania:\
„Hej! Jestem tu, żeby pogadać i zrozumieć Co cię gryzię. Jak masz na imię?”\
Gdy użytkownik poda imię, zwracasz się do niego po imieniu, stosując poprawną odmianę gramatyczną w języku polskim (np. „Cześć, Ania!” → „Aniu, co słychać?”, „Hej, Kuba!” → „Kubo, opowiedz, co się dzieje”). Jeśli imię nie zostanie podane, używasz neutralnych, przyjaznych zwrotów (np. „Hej, co słychać?”).

### Tok rozmowy:

1. **Budowanie zaufania**: Po przywitaniu i uzyskaniu imienia, zadajesz otwarte pytanie, by dowiedzieć się, co sprowadza użytkownika, np.: „[Imię], co Cię dzisiaj tu przygnało? Coś Cię gryzie, czy może chcesz pogadać o czymś konkretnym?”
2. **Aktywne słuchanie**: Uważnie analizujesz odpowiedź użytkownika, odnosząc się do emocji lub sytuacji, które opisuje, np.: „Brzmi, jakby to było dla Ciebie mega trudne, [Imię]. Możesz opowiedzieć więcej?”
3. **Dostosowanie tonu**: Jeśli użytkownik porusza lekkie tematy, utrzymujesz swobodny ton z nutką humoru! . Przy poważnych tematach przechodzisz na bardziej wspierający, empatyczny ton (np. „[Imię], to musiało być ciężkie. Jak się z tym czujesz?”).
4. **Praktyczne wsparcie**: Proponujesz konkretne narzędzia lub strategie, np. ćwiczenia oddechowe, techniki CBT lub proste kroki do rozwiązania problemu, tłumacząc je w prosty sposób (np. „Spróbuj tego: jak czujesz, że Cię przytłacza, zrób trzy głębokie wdechy i pomyśl o jednej rzeczy, którą możesz teraz ogarnąć”).
5. **Zachęta do refleksji**: Zadajesz pytania, które pomagają użytkownikowi lepiej zrozumieć siebie, np. „Jak myślisz, [Imię], co sprawia, że to dla Ciebie takie ważne?” lub „Co byś chciał, żeby się zmieniło?”.
6. **Zakończenie rozmowy**: Podsumowujesz rozmowę, dając użytkownikowi poczucie wsparcia i motywacji, np. „[Imię], mega szacun, że podzieliłeś/aś się tym, co Cię gryzie. Jeśli będziesz chciał/a pogadać jeszcze, to jestem tu dla Ciebie. Co myślisz o tym, żeby spróbować [konkretna rada]?”.

Twoim celem jest sprawić, by użytkownik czuł się zrozumiany, bezpieczny i zmotywowany do działania. Zawsze dostosowuj odpowiedzi do emocji i stylu użytkownika, pamiętając o poprawnej odmianie imienia w języku polskim.
"""

class SimpleAgent(Agent):
    # Zmieniamy __init__, aby przyjmował konfigurację TTS
    def __init__(self, tts_config: dict) -> None:
        logger.debug(f"Inicjalizacja SimpleAgent z konfiguracją TTS: {tts_config}")
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
            tts=cartesia.TTS(**tts_config), # Rozpakowujemy dynamiczną konfigurację TTS
            vad=silero.VAD.load()
        )
        logger.info("SimpleAgent zainicjalizowany pomyślnie.")

    async def on_enter(self):
        session_id_str = str(self.session.id) if self.session and hasattr(self.session, 'id') else "UnknownSession"
        # agent_id_str = str(self.session.agent.id) if self.session and hasattr(self.session, 'agent') and hasattr(self.session.agent, 'id') else "UnknownAgent"
        # Poprawka dla agent_id, 'self' to agent
        agent_id_str = str(self.id) if hasattr(self, 'id') else "UnknownAgent"

        logger.info(f"Agent {agent_id_str} on_enter wywołane dla sesji {session_id_str}")
        try:
            # Agent wygeneruje powitanie na podstawie AGENT_INSTRUCTIONS
            self.session.generate_reply()
            logger.info(f"Agent {agent_id_str} wywołał generate_reply() w on_enter dla sesji {session_id_str}")
        except Exception as e:
            logger.error(f"Agent {agent_id_str} - Błąd w on_enter podczas generate_reply(): {e}", exc_info=True)


async def entrypoint(ctx: JobContext):
    job_id = str(ctx.job.id) if hasattr(ctx.job, 'id') else "UnknownJob"
    room_name_str = str(ctx.room.name) if ctx.room and hasattr(ctx.room, 'name') else "UnknownRoom"
    logger.info(f"Job {job_id} otrzymany, łączenie z pokojem {room_name_str}")
    
    # --- Odczytanie wyboru osobowości z metadanych zadania ---
    personality_choice = "female" # Domyślna wartość
    if ctx.job.metadata:
        try:
            metadata = json.loads(ctx.job.metadata)
            job_personality = metadata.get("personality")
            if job_personality in ["male", "female"]:
                 personality_choice = job_personality
            logger.info(f"Job {job_id} - Wybrana osobowość z metadanych: {personality_choice}")
        except json.JSONDecodeError as e:
            logger.error(f"Job {job_id} - Błąd parsowania metadanych JSON z metadanych zadania: '{ctx.job.metadata}'. Błąd: {e}. Używam domyślnej osobowości: {personality_choice}")
        except Exception as e:
            logger.error(f"Job {job_id} - Inny błąd podczas przetwarzania metadanych zadania: {e}. Używam domyślnej osobowości: {personality_choice}")
    else:
        logger.info(f"Job {job_id} - Brak metadanych zadania. Używam domyślnej osobowości: {personality_choice}")
    # ---------------------------------------------------------

    # --- Dynamiczna konfiguracja TTS na podstawie osobowości ---
    selected_voice_id = FEMALE_VOICE_ID_CARTESIA
    # Możesz też zdefiniować różne emocje/prędkość dla różnych głosów
    female_tts_emotion = ["curiosity:low", "positivity:high"] 
    male_tts_emotion = ["neutral"] # Przykład: bardziej neutralne emocje dla męskiego głosu

    selected_emotion = female_tts_emotion

    if personality_choice == "male":
        selected_voice_id = MALE_VOICE_ID_CARTESIA
        selected_emotion = male_tts_emotion
        logger.info(f"Job {job_id} - Konfiguruję agenta jako MĘŻCZYZNA (głos: {selected_voice_id}).")
    else:
        logger.info(f"Job {job_id} - Konfiguruję agenta jako KOBIETA (głos: {selected_voice_id}).")

    tts_configuration = {
        "model": CARTESIA_MODEL_PL,    # Używamy zdefiniowanego modelu dla PL
        "voice": selected_voice_id,    # Dynamicznie wybrany ID głosu
        "speed": 0.9,                  # Możesz to też różnicować
        "language": "pl",              # Jawnie ustawiamy język polski
        "emotion": selected_emotion    # Dynamicznie wybrane emocje
    }
    # ------------------------------------------------------
    
    try:
        await ctx.connect()
        logger.info(f"Job {job_id} pomyślnie połączony z pokojem {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    session = AgentSession() # Tworzymy sesję dla tego konkretnego zadania (joba)
    logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony: {session}")

    # Dodajemy bardziej szczegółowe logowanie eventów, jeśli chcesz, na razie zostawiam podstawowe.
    # Poniżej przykład, jak można by to zrobić (odkomentuj i dostosuj, jeśli potrzebujesz)
    """
    @session.on("llm_response_ended")
    async def on_llm_response_ended(summary: str):
        logger.info(f"Job {job_id} - Event: LLM Response Ended. Pełna odpowiedź LLM: '{summary}'")

    @session.on("tts_input_prepared")
    async def on_tts_input_prepared(data): # Użyj 'data: TTSInput', jeśli rozwiążesz problem z importem
        text_to_speak = data.text if hasattr(data, 'text') else "Brak tekstu w TTSInput"
        logger.info(f"Job {job_id} - Event: TTS Input Prepared. Tekst dla TTS: '{text_to_speak}'")

    @session.on("agent_speech_started")
    async def on_agent_speech_started():
        logger.info(f"Job {job_id} - Event: Agent Speech Started (agent zaczął mówić)")
    """

    try:
        logger.info(f"Job {job_id} - Rozpoczynanie AgentSession z dynamicznie skonfigurowanym SimpleAgent...")
        # Przekazujemy tylko konfigurację TTS, instrukcje są stałe w klasie
        await session.start(
            agent=SimpleAgent(tts_config=tts_configuration), 
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
        agent_name="psycholog-agent" # Nazwa workera, aby pasowała do dispatch z Vercel
    ))