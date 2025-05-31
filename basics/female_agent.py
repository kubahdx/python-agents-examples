import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import Agent, AgentSession # Upewnij się, że Agent i AgentSession są importowane
from livekit.plugins import openai, silero, deepgram, cartesia

# ----- Konfiguracja Logowania na Początku Pliku -----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG)
logger = logging.getLogger("female-voice-agent") # Nazwa loggera dla agenta żeńskiego
logger.setLevel(logging.DEBUG)
# ----------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
logger.info("Załadowano zmienne środowiskowe z .env, jeśli plik istnieje (agent żeński).")

class SimpleAgent(Agent):
    def __init__(self) -> None:
        logger.debug("Inicjalizacja SimpleAgent (głos żeński)...")
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
            tts=cartesia.TTS(
                model="sonic-2-2025-05-08",
                voice="575a5d29-1fdc-4d4e-9afa-5a9a71759864", # <--- GŁOS ŻEŃSKI
                speed="slow",
                language="pl",
                emotion=["curiosity:low", "positivity:high", "surprise:high"]
            ),
            vad=silero.VAD.load()
        )
        logger.info("SimpleAgent (głos żeński) zainicjalizowany pomyślnie.")

    async def on_enter(self):
        session_id_str = str(self.session.id) if self.session and hasattr(self.session, 'id') else "UnknownSession"
        agent_id_str = str(self.session.agent.id) if self.session and hasattr(self.session, 'agent') and hasattr(self.session.agent, 'id') else "UnknownAgent"
        logger.info(f"Agent żeński {agent_id_str} on_enter wywołane dla sesji {session_id_str}")
        try:
            # Zamiast generate_reply() od razu, zaczekaj na interakcję użytkownika lub wyślij powitanie.
            # Możesz użyć self.session.say() do wysłania wiadomości powitalnej.
            # self.session.generate_reply() # Możesz to wywołać później, np. po pierwszej wiadomości od użytkownika.
            # Dla przykładu, wyślijmy powitanie:
            await self.session.say("Cześć! Jestem Twoją wirtualną kumpelą. Jak mogę Ci dzisiaj pomóc?", allow_interruptions=True)
            logger.info(f"Agent żeński {agent_id_str} wysłał powitanie w on_enter dla sesji {session_id_str}")
        except Exception as e:
            logger.error(f"Agent żeński {agent_id_str} - Błąd w on_enter: {e}", exc_info=True)


async def entrypoint(ctx: JobContext):
    job_id = str(ctx.job.id) if hasattr(ctx.job, 'id') else "UnknownJob"
    room_name_str = str(ctx.room.name) if ctx.room and hasattr(ctx.room, 'name') else "UnknownRoom"
    logger.info(f"Job {job_id} (agent żeński) otrzymany, łączenie z pokojem {room_name_str}")

    try:
        await ctx.connect()
        logger.info(f"Job {job_id} (agent żeński) pomyślnie połączony z pokojem {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} (agent żeński) - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    try:
        session = AgentSession() # Tworzenie sesji agenta
        logger.debug(f"Job {job_id} (agent żeński) - Obiekt AgentSession stworzony: {session}")
        logger.info(f"Job {job_id} (agent żeński) - Rozpoczynanie AgentSession z SimpleAgent (głos żeński)...")
        await session.start(
            agent=SimpleAgent(), # Przekazanie instancji agenta
            room=ctx.room
        )
        logger.info(f"Job {job_id} (agent żeński) - AgentSession wystartowała pomyślnie dla pokoju {room_name_str}")
    except Exception as e:
        logger.error(f"Job {job_id} (agent żeński) - Krytyczny błąd podczas inicjalizacji lub startu AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} (agent żeński) - Kończenie entrypoint.")


if __name__ == "__main__":
    logger.info("Uruchamianie aplikacji CLI dla workera agenta (głos żeński)...")
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="agent_female_nazwa"  # Twoja unikalna nazwa dla agenta żeńskiego
    ))
