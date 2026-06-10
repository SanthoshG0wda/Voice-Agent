import os
import logging
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
)

from livekit.plugins import nvidia, openai, silero

load_dotenv()

logging.basicConfig(level=logging.INFO)


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(
        vad=silero.VAD.load(),

        stt=nvidia.STT(),

        llm=openai.LLM(
            model="meta/llama-3.1-8b-instruct",
            api_key=os.environ["NVIDIA_API_KEY"],
            base_url="https://integrate.api.nvidia.com/v1",
        ),

        tts=nvidia.TTS(),
    )

    # Print conversation events
    @session.on("conversation_item_added")
    def on_conversation_item(event):
        item = event.item

        if item.role == "user":
            print(f"\n👤 USER: {item.content[0]}")

        elif item.role == "assistant":
            print(f"\n🤖 AGENT: {item.content[0]}")

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="""
            Your name is Agent Netovid
            You are a helpful voice assistant.
            Keep responses short and conversational.
            """
        ),
    )

    print("✅ Agent started and listening...")

    # Initial test message
    await session.generate_reply(
        instructions="Introduce yourself in one sentence."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )