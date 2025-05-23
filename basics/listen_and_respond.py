import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession # Usunięte LLMInput, TTSInput itp. na razie
from livekit.plugins import openai, silero, deepgram, cartesia

# ----- Konfiguracja Logowania na Początku Pliku -----
# Ustawiamy bardziej szczegółowy poziom logowania dla biblioteki livekit.agents
# oraz dla naszego własnego loggera.
logging.basicConfig(
    level=logging.INFO, # Ogólny poziom dla roota, można zmienić na DEBUG
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
# Aby zobaczyć więcej szczegółów z biblioteki livekit.agents:
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)
# Logger dla naszego skryptu
logger = logging.getLogger("listen-and-respond")
logger.setLevel(logging.DEBUG) # Ustawiamy na DEBUG, aby widzieć nasze komunikaty
# ----------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
logger.info("Załadowano zmienne środowiskowe z .env, jeśli plik istnieje.")

class SimpleAgent(Agent):
    def __init__(self) -> None:
        logger.debug("Inicjalizacja SimpleAgent...") # DEBUG log
        super().__init__(
            instructions=r"""
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
            """, # Dodano 'r' przed """ dla raw string
            stt=deepgram.STT(
                model="nova-2-general", # Poprzednio było "nova-2", "nova-2-general" jest bardziej jawne
                language="pl",
                interim_results=True,
                smart_format=True,
                punctuate=True,
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(
                model="sonic-2-2025-05-08", 
                voice="3d335974-4c4a-400a-84dc-ebf4b73aada6",
                speed=0.9,
                language="pl",
                emotion=["curiosity:high", "positivity:high", "surprise:high"] # Możesz to odkomentować, jeśli chcesz
            ),
            vad=silero.VAD.load()
        )
        logger.info("SimpleAgent zainicjalizowany pomyślnie.") # INFO log

    async def on_enter(self):
        session_id_str = str(self.session.id) if self.session and hasattr(self.session, 'id') else "UnknownSession"
        agent_id_str = str(self.session.agent.id) if self.session and hasattr(self.session, 'agent') and hasattr(self.session.agent, 'id') else "UnknownAgent"
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

    try:
        await ctx.connect()
        logger.info(f"Job {job_id} pomyślnie połączony z pokojem {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    # Tworzenie sesji wewnątrz bloku try, aby złapać ewentualne błędy inicjalizacji
    try:
        session = AgentSession()
        logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony: {session}")

        # --- Możemy dodać bardzo proste logowanie transkrypcji tutaj, jeśli STT działa ---
        # To jest bardziej zaawansowane i wymagałoby dostępu do strumienia audio
        # Na razie polegamy na logach z livekit.agents.stt i livekit.agents.llm

        logger.info(f"Job {job_id} - Rozpoczynanie AgentSession z SimpleAgent...")
        await session.start(
            agent=SimpleAgent(),
            room=ctx.room
        )
        logger.info(f"Job {job_id} - AgentSession wystartowała pomyślnie dla pokoju {room_name_str}")

    except Exception as e:
        logger.error(f"Job {job_id} - Krytyczny błąd podczas inicjalizacji lub startu AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} - Kończenie entrypoint. AgentSession (jeśli wystartowała) zakończy pracę, gdy pokój zostanie zamknięty lub agent się rozłączy.")


if __name__ == "__main__":
    logger.info("Uruchamianie aplikacji CLI dla workera agenta...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))