import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.voice import (
    Agent,
    AgentSession,
    LLMInput,
    TTSInput,
    VADInput,
    STTInput, # Dodany import
)
from livekit.plugins import openai, silero, deepgram, cartesia
from livekit import rtc # Potrzebne dla typów w eventach

# ----- Konfiguracja Logowania na Początku Pliku -----
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)
logging.getLogger("livekit.agents").setLevel(logging.DEBUG) 
logger = logging.getLogger("listen-and-respond")
logger.setLevel(logging.DEBUG) 
# ----------------------------------------------------

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')
logger.info("Załadowano zmienne środowiskowe z .env")

class SimpleAgent(Agent):
    def __init__(self) -> None:
        logger.debug("Inicjalizacja SimpleAgent...")
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
            """,
            stt=deepgram.STT(
                model="nova-2", # W dokumentacji Deepgram nova-2 jest jako nova-2-general
                language="pl",
                interim_results=True,
                smart_format=True,
                punctuate=True,
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=cartesia.TTS(
                model="sonic-2", # Upewnij się, że ten model Cartesia dobrze działa z polskim
                voice="3d335974-4c4a-400a-84dc-ebf4b73aada6",
                speed=0.9,
                emotion=["curiosity:medium", "positivity:high"]
                # language="pl" # Sprawdź, czy Cartesia TTS wymaga tego parametru dla j. polskiego
            ),
            vad=silero.VAD.load()
        )
        logger.info("SimpleAgent zainicjalizowany pomyślnie.")

    async def on_enter(self):
        session_id = self.session.id if self.session else "UnknownSession"
        agent_id = self.session.agent.id if self.session and self.session.agent else "UnknownAgent"
        logger.info(f"Agent {agent_id} on_enter wywołane dla sesji {session_id}")
        # Zgodnie z Twoim kodem, agent generuje odpowiedź (prawdopodobnie powitalną) zaraz po wejściu
        self.session.generate_reply()
        logger.info(f"Agent {agent_id} wywołał generate_reply() w on_enter dla sesji {session_id}")


async def entrypoint(ctx: JobContext):
    job_id = ctx.job.id
    room_name = ctx.room.name if ctx.room else "UnknownRoom"
    logger.info(f"Job {job_id} otrzymany, łączenie z pokojem {room_name}")
    
    try:
        await ctx.connect()
        logger.info(f"Job {job_id} połączony z pokojem {room_name}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas ctx.connect(): {e}", exc_info=True)
        return

    session = AgentSession()
    logger.debug(f"Job {job_id} - Obiekt AgentSession stworzony: {session}")

    # --- Dodajemy Nasłuchiwanie na Zdarzenia AgentSession ---
    @session.on("vad_input_started")
    async def on_vad_input_started(event: VADInput):
        logger.debug(f"Job {job_id} - Event: VAD Input Started - Speaking: {event.speaking}")

    @session.on("vad_input_ended")
    async def on_vad_input_ended(event: VADInput):
        logger.debug(f"Job {job_id} - Event: VAD Input Ended - Speaking: {event.speaking}")
    
    @session.on("stt_input_started")
    async def on_stt_input_started(event: STTInput):
        logger.debug(f"Job {job_id} - Event: STT Input Started dla strumienia: {event.stream_id}")

    @session.on("stt_input_text_stream_started")
    async def on_stt_text_stream_started():
        logger.debug(f"Job {job_id} - Event: STT Input Text Stream Started")

    @session.on("stt_input_text_stream_updated")
    async def on_stt_text_stream_updated(text: str):
        logger.debug(f"Job {job_id} - Event: STT Input Text Stream Updated - Interim Text: '{text}'")

    @session.on("stt_input_text_stream_ended")
    async def on_stt_text_stream_ended(text: str):
        logger.info(f"Job {job_id} - Event: STT Input Text Stream Ended - Final Text: '{text}'")

    @session.on("llm_input_text_needed")
    async def on_llm_input_text_needed():
        logger.info(f"Job {job_id} - Event: LLM Input Text Needed (agent przygotowuje dane dla LLM)")

    @session.on("llm_input_prepared")
    async def on_llm_input_prepared(data: LLMInput):
        logger.info(f"Job {job_id} - Event: LLM Input Prepared. Chat history length: {len(data.chat_history.messages)}")
        if data.chat_history.messages:
            if data.chat_history.messages[-1].role == "user":
                 logger.info(f"Job {job_id} - Ostatnia wiadomość użytkownika dla LLM: '{data.chat_history.messages[-1].content}'")
            elif data.chat_history.messages[-1].role == "assistant":
                 logger.info(f"Job {job_id} - Ostatnia wiadomość asystenta dla LLM: '{data.chat_history.messages[-1].content}'")
        else:
            logger.info(f"Job {job_id} - Historia czatu dla LLM jest pusta.")


    @session.on("llm_response_started")
    async def on_llm_response_started(text_stream):
        logger.info(f"Job {job_id} - Event: LLM Response Started (LLM zaczął generować odpowiedź)")
        # Możesz odkomentować poniższe, aby logować fragmenty odpowiedzi LLM, ale może być tego dużo
        # async for chunk in text_stream:
        #     logger.debug(f"Job {job_id} - LLM response chunk: {chunk}")

    @session.on("llm_response_ended")
    async def on_llm_response_ended(summary: str):
        logger.info(f"Job {job_id} - Event: LLM Response Ended. Pełna odpowiedź LLM: '{summary}'")

    @session.on("tts_input_text_needed")
    async def on_tts_input_text_needed():
        logger.info(f"Job {job_id} - Event: TTS Input Text Needed (agent przygotowuje tekst dla TTS)")

    @session.on("tts_input_prepared")
    async def on_tts_input_prepared(data: TTSInput):
        logger.info(f"Job {job_id} - Event: TTS Input Prepared. Tekst dla TTS: '{data.text}'")


    @session.on("tts_audio_stream_started")
    async def on_tts_audio_stream_started(audio_stream):
        logger.info(f"Job {job_id} - Event: TTS Audio Stream Started (TTS zaczął generować audio)")

    @session.on("tts_audio_stream_ended")
    async def on_tts_audio_stream_ended():
        logger.info(f"Job {job_id} - Event: TTS Audio Stream Ended")
        
    @session.on("agent_speech_started")
    async def on_agent_speech_started():
        logger.info(f"Job {job_id} - Event: Agent Speech Started (agent zaczął mówić)")

    @session.on("agent_speech_finished")
    async def on_agent_speech_finished(interrupted: bool):
        logger.info(f"Job {job_id} - Event: Agent Speech Finished. Interrupted: {interrupted}")
    
    @session.on("session_ended") # Dodatkowe zdarzenie
    async def on_session_ended(error: Exception | None):
        if error:
            logger.error(f"Job {job_id} - Event: AgentSession Ended with error: {error}", exc_info=error)
        else:
            logger.info(f"Job {job_id} - Event: AgentSession Ended successfully.")

    try:
        logger.info(f"Job {job_id} - Rozpoczynanie AgentSession...")
        await session.start(
            agent=SimpleAgent(),
            room=ctx.room
        )
        logger.info(f"Job {job_id} - AgentSession wystartowała pomyślnie dla pokoju {ctx.room.name if ctx.room else 'UnknownRoom'}")
    except Exception as e:
        logger.error(f"Job {job_id} - Błąd podczas uruchamiania AgentSession: {e}", exc_info=True)
    finally:
        logger.info(f"Job {job_id} - Kończenie entrypoint. AgentSession zakończy pracę, gdy pokój zostanie zamknięty lub agent się rozłączy.")


if __name__ == "__main__":
    logger.info("Uruchamianie aplikacji CLI dla workera agenta...")
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))