import os
import sqlite3
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph, END
from langgraph.prebuilt import ToolNode
from langsmith import Client, evaluate

load_dotenv()

if os.environ.get("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "exercise-4-agent-with-tools"

if "TAVILY_API_KEY" not in os.environ:
    raise ValueError("TAVILY_API_KEY not found. Check your .env file.")

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("OPENAI_API_KEY not found. Check your .env file.")

@tool
def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: first int
        b: second int
    """
    return a / b


@tool
def search_web(query: str) -> str:
    """Search the web for current information using Tavily.

    Args:
        query: question or search query
    """

    try:
        import json
        import subprocess

        payload = json.dumps({
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "max_results": 3,
            "search_depth": "advanced",
        })

        result = subprocess.run(
            ["curl", "-s", "https://api.tavily.com/search",
             "-X", "POST",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )

        if result.returncode != 0:
            return f"Tavily search failed: curl error: {result.stderr}"

        response = json.loads(result.stdout)
        search_docs = response.get("results", [])
    except Exception as error:
        return f"Tavily search failed: {error}"

    if not search_docs:
        return "No useful Tavily search results were found."

    formatted_search_docs = "\n\n---\n\n".join(
        (f'<Document href="{doc.get("url", "")}">\n'
            f'Title: {doc.get("title", "")}\n'
            f'{doc.get("content", "")}\n'
            f"</Document>") for doc in search_docs)

    return formatted_search_docs


conn = sqlite3.connect(":memory:", check_same_thread=False)
conn.execute("""CREATE TABLE todos (id INTEGER PRIMARY KEY,
                                    task TEXT NOT NULL,
                                    deadline TEXT)""")

conn.executemany("""INSERT INTO todos (task, deadline) VALUES (?, ?)""",
                 [("Finish booking travel to Hong Kong",
                   "End of next week"),
                  ("Call parents back about Thanksgiving plans",
                   None),
                  ("Drop by Yoga in person",
                   "Sunday")])
conn.commit()


@tool
def database_lookup(query: str) -> str:
    """Execute one CRUD SQL statement against the SQLite database.

    Supported operations: SELECT, INSERT, UPDATE, and DELETE.

    Args:
        query: one complete SQLite statement
    """

    query = query.strip()

    if not query:
        return "SQL_ERROR: The query cannot be empty."

    operation = query.split(maxsplit=1)[0].upper()

    allowed_operations = {"SELECT", "INSERT", "UPDATE", "DELETE"}

    if operation not in allowed_operations:
        return ("SQL_ERROR: Only SELECT, INSERT, UPDATE, "
                "and DELETE statements are allowed.")

    try:
        cursor = conn.execute(query)

        if operation == "SELECT":
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]

            if not rows:
                return "The query returned no records."

            formatted_rows = []

            for row in rows:
                values = [f"{column}: {value}"
                          for column, value in zip(columns, row)]

                formatted_rows.append("- " + ", ".join(values))
            return "\n".join(formatted_rows)

        conn.commit()

        return (f"{operation} completed successfully. "
                f"Rows affected: {cursor.rowcount}")

    except sqlite3.Error as error:
        conn.rollback()
        return f"SQL_ERROR: {error}"


tools = [add, multiply, divide, search_web, database_lookup]

llm = ChatOpenAI(model="gpt-4o-mini",
               temperature=0,
               max_retries=3)

llm_with_tools = llm.bind_tools(tools,
                                tool_choice="auto",
                                parallel_tool_calls=False)


sys_msg = SystemMessage(
    content=("You are a helpful assistant with access to these tools: "
             "add, multiply, divide, search_web, and database_lookup. "

             "The SQLite database has this schema:\n"
             "todos(\n"
             "    id INTEGER PRIMARY KEY,\n"
             "    task TEXT NOT NULL,\n"
             "    deadline TEXT\n"
             ")\n"

             "Use database_lookup whenever the user wants to read or modify "
             "database data. Translate the user's request into one valid "
             "SQLite SELECT, INSERT, UPDATE, or DELETE statement. "
             "Never invent tables or columns. "
             "Before UPDATE or DELETE, ensure that the WHERE condition "
             "matches the user's requested scope. "
             "Use SELECT when the user asks to view, list, find, count, "
             "filter, or summarize database records. "

             "Select the appropriate tool for the user's request. "
             "Call only one tool at a time. After receiving a tool result, "
             "decide whether another available tool is necessary. "
             "When all necessary tools have been used, provide the final answer."))


def assistant(state: MessagesState):
    """Select the appropriate tool or produce the final answer."""

    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    return {"messages": [response]}


def route_tool_call(state: MessagesState) -> str:
    """Route to the specific tool requested by the assistant."""

    last_message = state["messages"][-1]

    if not last_message.tool_calls:
        return "end"

    tool_name = last_message.tool_calls[0]["name"]

    routes = {"add": "add_tool",
              "multiply": "multiply_tool",
              "divide": "divide_tool",
              "search_web": "search_web_tool",
              "database_lookup": "database_lookup_tool"}

    return routes.get(tool_name, "end")


builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)

builder.add_node("add_tool", ToolNode([add]))
builder.add_node("multiply_tool", ToolNode([multiply]))
builder.add_node("divide_tool", ToolNode([divide]))
builder.add_node("search_web_tool", ToolNode([search_web]))
builder.add_node("database_lookup_tool", ToolNode([database_lookup]))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", route_tool_call, {
        "add_tool": "add_tool",
        "multiply_tool": "multiply_tool",
        "divide_tool": "divide_tool",
        "search_web_tool": "search_web_tool",
        "database_lookup_tool": "database_lookup_tool",
        "end": END})

for tool_node in ["add_tool",
                  "multiply_tool",
                  "divide_tool",
                  "search_web_tool",
                  "database_lookup_tool"]:
    builder.add_edge(tool_node, "assistant")

graph = builder.compile()