from dotenv import load_dotenv
import getpass
import os
from src.prompt import *
from src.helper import *
import asyncio
import chainlit as cl

load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("MISTRAL_API_KEY")
_set_if_undefined("GOOGLE_API_KEY")

research_supervisor_node = make_supervisor_node(llm, ["search", "web_scraper"])
research_builder = StateGraph(State)
research_builder.add_node("supervisor", research_supervisor_node)
research_builder.add_node("search", search_node)
research_builder.add_node("web_scraper", web_scraper_node)
research_builder.add_edge(START, "supervisor")
research_graph = research_builder.compile()

doc_writing_supervisor_node = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)

paper_writing_builder = StateGraph(State)
paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
paper_writing_builder.add_node("doc_writer", doc_writing_node)
paper_writing_builder.add_node("note_taker", note_taking_node)
paper_writing_builder.add_node("chart_generator", chart_generating_node)

paper_writing_builder.add_edge(START, "supervisor")
paper_writing_graph = paper_writing_builder.compile()

teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

def call_research_team(state: State) -> Command[Literal["supervisor"]]:
    response = research_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="research_team"
                )
            ]
        },
        goto="supervisor",
    )


def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
    response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="writing_team"
                )
            ]
        },
        goto="supervisor",
    )

# Define the graph.
super_builder = StateGraph(State)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("research_team", call_research_team)
super_builder.add_node("writing_team", call_paper_writing_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

@cl.on_message
async def main(message: cl.Message):
    final_answer = None

    async with cl.Step(name="Agent Execution Trace", show_input=False) as step:
        step.output = ""  # initialize

        for s in super_graph.stream(
            {
                "messages": [
                    ("user", message.content)
                ],
            },
            {"recursion_limit": 150},
        ):
            # Append logs inside Chainlit UI
            step.output += f"{s}\n\n"
            await step.update()

            # Non-blocking throttle
            await asyncio.sleep(12)

            # Capture final answer
            if "research_team" in s:
                msg = s["research_team"]["messages"][-1].content
                final_answer = msg[0]["text"] if isinstance(msg, list) else msg

    # Send ONLY final answer to main chat
    if final_answer:
        await cl.Message(content=final_answer).send()
    else:
        await cl.Message(content="No final answer generated.").send()