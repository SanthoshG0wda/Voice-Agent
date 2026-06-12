import logging
from typing import Optional
from urllib.parse import quote
import aiohttp
import asyncio
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    RunContext,
    ToolError,
    cli,
    function_tool,
    inference,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import (
    ai_coustics,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent-Christy")

load_dotenv(".env.local")


class DefaultAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""\"\"\"
            Your name is Christy.
            You are the official AI teacher and assistant for Christ King Institution.
            Your role is to help students, parents, and staff with accurate information about the school.
            Whenever you need specific details about admissions, events, or school policies, use your `search_school_website` tool to look it up.
            Be warm, knowledgeable, encouraging, and highly professional, just like a great teacher.
            Keep your spoken responses short, engaging, and conversational.
            \"\"\"""",
            tools=[EndCallTool(
                extra_description="""""",
                end_instructions="""Thank the user for their time and say goodbye.""",
                delete_room=False,
            )],
        )
    async def on_enter(self):
        await self.session.generate_reply(
            instructions="""I am Christy, the Christ King Institution AI teacher. How can I help you today?""",
            allow_interruptions=True,
        )
    @function_tool(name="Get_school_info")
    async def _http_tool_Get_school_info(
        self, context: RunContext, contact_info: str, courses: str, facilities: str, achievements: str, admissions: str, about: str
    ) -> str | None:
        """
        Search the Christ King Institution website (christkinginstitution.com) for information about the school, admissions, academics, or events.

        Args:
            contact_info: if user ask about school contact info
            courses: if user ask about school courses available 
            facilities: if user ask about school facilities
            achievements: if user ask about school achievements
            admissions: if user ask about school admissions
            about: if user ask about the school 
        """

        url = "https://www.christkinginstitution.com/"
        payload = {
            k: v for k, v in {
                "contact_info": contact_info,
                "courses": courses,
                "facilities": facilities,
                "achievements": achievements,
                "admissions": admissions,
                "about": about,
            }.items() if v is not None
        }

        try:
            session = utils.http_context.http_session()
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout, params=payload) as resp:
                if resp.status >= 400:
                    raise ToolError(f"error: HTTP {resp.status}")
                return await resp.text()
        except ToolError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            raise ToolError(f"error: {e!s}") from e


server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

server.setup_fnc = prewarm

@server.rtc_session(agent_name="Christy")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="en"),
        llm=inference.LLM(
            model="openai/gpt-5.1-chat-latest",
            extra_kwargs={"reasoning_effort": "low"},
        ),
        tts=inference.TTS(
            model="cartesia/sonic-3",
            voice="a167e0f3-df7e-4d52-a9c3-f949145efdab",
            language="en-US"
        ),
        turn_handling=TurnHandlingOptions(turn_detection=MultilingualModel()),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=DefaultAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_L,
                ),
            ),
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)
