from openai import OpenAI


def generate_question(path_str, start, end):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    prompt = """
    You are an expert in Knowledge Graph Multi-Hop Question Generation. Please generate a natural language question based on the provided path information, starting from the source node and following the logical flow of the path.
    
    **Path Format Description:**
    The path consists of a sequence of "Node-Relation->Node" associations. For example, "A -relation1-> B -relation2-> C" indicates connecting from A to B via relation1, and then to C via relation2.
    
    **Generation Requirements:**
    1. **Single-Sentence Question:** The question must be a single, coherent sentence. Strictly avoid splitting it into multiple sub-questions (e.g., do not use structures like "Who is...? And where was he...?").
    2. **Coreference Resolution:** Do not mention the names of intermediate nodes in the question. Instead, refer to them using a description combining the "Start Node + Relation".
    3. **Logical Nesting:** Convert preceding path nodes into modifiers (attributives) for subsequent nodes to form a nested query.
    4. **Unique Answer:** The final answer to the question must be the [End Node]. The name of the [End Node] must NOT appear in the question.
    
    **Example:**
    Input:
    - Path: Mona Lisa -(created by)-> Da Vinci -(born in)-> Italy
    - Start Node: Mona Lisa
    - End Node: Italy
    ❌ Incorrect Output: Who created the Mona Lisa? Where was he born?
    ✅ Correct Output: In which country was the creator of the Mona Lisa born?
    
    **Now please process the following:**
    Input Format:
    - Path: [Specific association path]
    - Start Node: [Start Node]
    - End Node: [Target Answer]
    """

    response = client.chat.completions.create(
        model="x-ai/grok-4.1-fast:free",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": f"Path:{path_str}\nStart Node:{start}\nEnd Node:{end}",
            },
        ],
        extra_body={"reasoning": {"enabled": False}},
    )

    return response.choices[0].message.content


print(
    generate_question(
        "Xiaomi Phone -(invented)-> Lei Jun -(based in)-> China -(capital)-> Beijing",
        "Xiaomi Phone",
        "Beijing",
    )
)
