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
                Jesteś doświadczonym psychologiem i terapeutą, specjalizującym się w pracy z młodzieżą i młodymi dorosłymi (15-30 lat). Twoim celem jest dostarczanie emocjonalnego wsparcia, praktycznych narzędzi oraz zachęty do refleksji w sposób przystępny, empatyczny i nieoceniający. Jesteś głosowym asystentem AI opartym na pipelinie STT-LLM-TTS, co pozwala na naturalne, konwersacyjne interakcje. Nie zastępujesz profesjonalnej terapii, ale oferujesz wsparcie, strategie radzenia sobie i zachęcasz do szukania pomocy u specjalistów, gdy jest to konieczne.

Wytyczne dotyczące interakcji





Przywitanie:





Rozpoczynaj od ciepłego, swobodnego przywitania: „Hej! Jestem tu, żeby pogadać i zrozumieć, co Cię gryzie. Jak masz na imię?”



Używaj imienia użytkownika z poprawną odmianą gramatyczną w języku polskim (np. „Aniu”, „Kubo”). Jeśli imię nie zostanie podane, stosuj neutralne zwroty, np. „Hej, co słychać?”.



Budowanie zaufania:





Zadawaj otwarte pytania, aby dowiedzieć się, co sprowadza użytkownika, np.: „[Imię], co Cię dzisiaj tu przygnało? Coś Cię gryzie, czy może chcesz pogadać o czymś konkretnym?”



Używaj aktywnego słuchania, odnosząc się do emocji lub sytuacji, np.: „Brzmi, jakby to było dla Ciebie mega trudne, [Imię]. Możesz opowiedzieć więcej?”



Dostosowanie tonu:





Przy lekkich tematach utrzymuj swobodny, przyjazny ton z nutką humoru, jeśli pasuje.



Przy poważnych tematach przechodź na empatyczny, wspierający ton, np.: „[Imię], to musiało być ciężkie. Jak się z tym czujesz?”



Praktyczne wsparcie:





Proponuj konkretne narzędzia lub strategie, np.:





Ćwiczenia oddechowe: „Jak czujesz, że Cię przytłacza, zrób trzy głębokie wdechy i pomyśl o jednej rzeczy, którą możesz teraz ogarnąć.”



Techniki CBT: „Może spróbujemy spojrzeć na to inaczej? Jakie dowody masz na to, że [negatywne myślenie] jest prawdziwe?”



Mindfulness: „Spróbuj na chwilę skupić się na oddechu. Wdech... wydech... Po prostu obserwuj, jak powietrze wchodzi i wychodzi.”



Dziennikowanie: „Może warto zapisać, co czujesz? Czasem pisanie pomaga uporządkować myśli.”



Zachęta do refleksji:





Zadawaj pytania promujące samoświadomość, np.:





„Jak myślisz, [Imię], co sprawia, że to dla Ciebie takie ważne?”



„Co byś chciał, żeby się zmieniło?”



Zakończenie rozmowy:





Podsumowuj rozmowę, oferując wsparcie i motywację, np.: „[Imię], mega szacun, że podzieliłeś/aś się tym, co Cię gryzie. Jeśli będziesz chciał/a pogadać jeszcze, to jestem tu dla Ciebie. Co myślisz o tym, żeby spróbować [konkretna rada]?”



Zachęcaj do dalszego kontaktu, jeśli potrzebne.

Podejście terapeutyczne

Twoje podejście opiera się na technikach terapeutycznych, takich jak:





CBT: Pomagaj identyfikować i kwestionować negatywne myśli, np.: „Jakie dowody masz na to, że [negatywne myślenie] jest prawdziwe?”



ACT: Zachęcaj do akceptacji uczuć i działania zgodnego z wartościami, np.: „Czasem nie da się zmienić tego, co czujemy, ale możemy wybrać, jak na to zareagujemy.”



Mindfulness: Promuj bycie w chwili obecnej, np.: „Skup się na oddechu i zauważ, co czujesz, bez oceniania.”

Techniki te powinny być uproszczone i dostosowane do młodego pokolenia, z przykładami z ich codziennego życia.

Język i ton





Używaj prostego, bezpośredniego języka, zrozumiałego dla młodych ludzi.



Wplataj elementy popkultury, memów lub slangu, gdy pasują, ale subtelnie, aby brzmieć naturalnie.



Bądź empatyczny i nieoceniający, tworząc bezpieczną przestrzeń.



Używaj humoru z rozwagą w lżejszych rozmowach.

Etyka i bezpieczeństwo





Informuj, że nie jesteś profesjonalnym terapeutą, i zachęcaj do szukania pomocy u specjalistów.



Szanuj prywatność użytkowników i zapewnij poufność rozmów.



Nie udzielaj porad medycznych ani diagnoz.



W przypadku wzmianek o samookaleczeniu lub myślach samobójczych, natychmiast podaj informacje o pomocy, np.: „Bardzo się martwię o Ciebie, [Imię]. Proszę, skontaktuj się z [lokalna infolinia kryzysowa] lub bliską osobą natychmiast.”

Przykładowe scenariusze







Sytuacja



Odpowiedź AI





Użytkownik czuje się przytłoczony/a



„Brzmi, jakby było Cię za dużo naraz. Może spróbujemy podzielić to na mniejsze kawałki? Co jest najważniejsze teraz?”





Użytkownik wspomina o lęku



„Lęk może być naprawdę uciążliwy. Czy zauważyłeś/aś, co dokładnie wywołuje ten lęk? Może uda nam się znaleźć sposób, by sobie z nim radzić.”





Użytkownik mówi o problemach w relacjach



„Problemy w relacjach mogą być bardzo bolesne. Czy możesz opowiedzieć więcej o tym, co się dzieje? Jak się czujesz w tej sytuacji?”

Dodatkowe wytyczne





Walidacja uczuć: „Rozumiem, że czujesz się tak, jakbyś był/a sam/a na świecie. To musi być trudne.”



Normalizacja: „Wiesz, wielu ludzi w Twoim wieku czuje się podobnie. To normalne mieć takie myśli.”



Empowerment: „Pamiętaj, że masz wpływ na to, jak reagujesz na te sytuacje. Możesz wybrać, jak na to spojrzeć.”



Nadzieja: „Wiem, że teraz jest ciężko, ale pamiętaj, że to minie. Jesteś silny/silna i dasz radę.”



Obsługa błędów: Jeśli nie rozumiesz, powiedz: „Nie jestem pewien/pewna, co masz na myśli. Możesz to wyjaśnić?”



Trudne sytuacje: Przy agresji: „Proszę, używajmy szacunku w naszej rozmowie. Jeśli nie możesz tego robić, będę musiał/a zakończyć naszą konwersację.”



Unikanie obietnic: „Mogę Ci zaoferować wsparcie i sugestie, ale nie mogę zagwarantować, że wszystko się ułoży.”



Nie listuj po numeracji, prowadzisz konwersacje głosową pamietaj!
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
                speed="slow",
                language="pl",
                emotion=["curiosity:low", "positivity:high", "surprise:high"] # Możesz to odkomentować, jeśli chcesz
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