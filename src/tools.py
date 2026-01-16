from langchain.tools import tool

# -----------------------
# Tools
# -----------------------

@tool("multiplier", description="Performs product calculations. Use this for calculating products. input -> int, int : output -> int")
def multiply(a: int, b: int) -> int:
    return a * b

@tool("adder", description="Performs sum calculations. Use this for calculating sum. input -> int, int : output -> int")
def add(a: int, b: int) -> int:
    return a + b

def get_tools():
    tools = [multiply, add]
    return tools