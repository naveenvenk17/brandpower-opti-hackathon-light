import asyncio
from src.agent.master_agent import get_agent_executor
from langchain_core.messages import AIMessage, HumanMessage

async def main():
    agent_executor = get_agent_executor()

    # Test 1: Simple greeting
    print("--- Testing: Simple greeting ---")
    chat_history = []
    user_message = "Hello, who are you?"
    response = await agent_executor.ainvoke({"input": user_message, "chat_history": chat_history})
    print(f"User: {user_message}")
    print(f"Agent: {response['output']}")
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response['output']))
    print("--- End Test ---")
    print()

    # Test 2: Forecast tool query with ambiguous brand and follow-up
    print("--- Testing: Forecast tool query with ambiguous brand and follow-up ---")
    user_message = "What is the baseline forecast for familia in colombia?"
    response = await agent_executor.ainvoke({"input": user_message, "chat_history": chat_history})
    print(f"User: {user_message}")
    print(f"Agent: {response['output']}")
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response['output']))

    # Simulate user selecting a brand
    user_message = "1" # Selecting 'familia budweiser'
    response = await agent_executor.ainvoke({"input": user_message, "chat_history": chat_history})
    print(f"User: {user_message}")
    print(f"Agent: {response['output']}")
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response['output']))
    print("--- End Test ---")
    print()

    # Test 3: Optimization tool query
    print("--- Testing: Optimization tool query ---")
    user_message = "I have a budget of 500000. Can you optimize my marketing allocation?"
    response = await agent_executor.ainvoke({"input": user_message, "chat_history": chat_history})
    print(f"User: {user_message}")
    print(f"Agent: {response['output']}")
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response['output']))
    print("--- End Test ---")
    print()

    # Test 4: Off-topic query
    print("--- Testing: Off-topic query ---")
    user_message = "What is the weather like today?"
    response = await agent_executor.ainvoke({"input": user_message, "chat_history": chat_history})
    print(f"User: {user_message}")
    print(f"Agent: {response['output']}")
    chat_history.append(HumanMessage(content=user_message))
    chat_history.append(AIMessage(content=response['output']))
    print("--- End Test ---")
    print()

if __name__ == "__main__":
    asyncio.run(main())