import json
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
    1. Single-Sentence Question: The question must be a single, coherent sentence. Strictly avoid splitting it into multiple sub-questions (e.g., do not use structures like "Who is...? And where was he...?").
    2. Coreference Resolution: Do not mention the names of intermediate nodes in the question. Instead, refer to them using a description combining the "Start Node + Relation".
    3. Logical Nesting: Convert preceding path nodes into modifiers (attributives) for subsequent nodes to form a nested query.
    4. Unique Answer: The final answer to the question must be the [End Node]. The name of the [End Node] must NOT appear in the question.

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
        model="openai/gpt-4o-mini",
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


def evaluate_question(triples_str, start_node, end_node, ref_question, gen_question):
    """
    使用 LLM 作为裁判，对生成的 KGQG 问题进行打分 (0-100)。

    Args:
        triples_str (str): 所有的三元组上下文 (包含噪音)
        start_node (str): 起点
        end_node (str): 终点 (答案)
        ref_question (str): 参考问题 (Gold)
        gen_question (str): 模型生成的问题 (Pred)

    Returns:
        int: 0-100 的整数分数
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

    prompt = """
    You are an expert evaluator for Knowledge Graph Question Generation (KGQG). Your task is to rate a generated question based on the provided Graph Context and Reference Question.

    **Input Data:**
    1. Graph Context: A set of triples (Subject, Relation, Object). NOTE: This context contains NOISE. The question should be based on a valid path within this context.
    2. Start Node: The topic entity of the question.
    3. End Node: The answer entity.
    4. Reference Question: The ground truth question.
    5. Generated Question: The question to be evaluated.

    **Evaluation Criteria (Total 100 points, 5 aspects, 20 points each):**

    1. Fluency & Grammar (0-20 pts): 
       - Is the generated question grammatically correct and natural? 
       - 20: Perfect; 0: Unreadable.

    2. Faithfulness (0-20 pts): 
       - Is the question supported by the Graph Context? 
       - Does it mention relations or entities that exist in the context? (Avoid hallucinations).
       - 20: Fully supported; 0: Hallucinated relations/entities.

    3. Logical Correctness (0-20 pts): 
       - Is the question answerable? Does the logic strictly lead from the [Start Node] to the [End Node]?
       - 20: Logic is sound and leads to the answer; 0: Logic is broken or leads to a wrong node.

    4. Constraints Compliance (0-20 pts): 
       - Does the question AVOID mentioning the [End Node] name explicitly? (Crucial!)
       - Is it a multi-hop question (not a simple one-hop query)?
       - 20: Multi-hop and Answer is hidden; 0: Answer leaked in question or single-hop.

    5. Semantic Alignment (0-20 pts): 
       - Does the generated question ask roughly the same thing as the Reference Question? (Even if phrased differently).
       - 20: Same intent; 0: Completely different topic.

    **Output Format:**
    You must output a strictly valid JSON object containing two fields:
    - "score": An integer between 0 and 100.
    - "reason": A brief explanation of the deduction.

    Example Output:
    {"score": 85, "reason": "Grammar is good. Logic is correct. However, the question is slightly ambiguous compared to the reference."}
    """

    user_content = f"""
    [Graph Context]:
    {triples_str}

    [Start Node]: {start_node}
    [End Node]: {end_node}

    [Reference Question]: {ref_question}
    [Generated Question]: {gen_question}
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},  # 强制 JSON 模式
            temperature=0.0  # 保持评估稳定性
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        return int(result_json.get("score", 0))

    except Exception as e:
        print(f"[LLM Eval Error]: {e}")
        # 如果出错，返回 -1 或者 0，视情况而定
        return 0


if __name__ == "__main__":
    print(
        generate_question(
            "Xiaomi Phone -(invented)-> Lei Jun -(based in)-> China -(capital)-> Beijing",
            "Xiaomi Phone",
            "Beijing",
        )
    )



    triples = """
    (Xiaomi Phone, invented_by, Lei Jun)
    (Lei Jun, based_in, China)
    (China, capital, Beijing)
    (Apple, competitor, Xiaomi Phone)
    """

    # Case 1: 完美生成
    score1 = evaluate_question(
        triples_str=triples,
        start_node="Xiaomi Phone",
        end_node="Beijing",
        ref_question="What is the capital of the country where the inventor of Xiaomi Phone is based?",
        gen_question="Which city is the capital of the place where Lei Jun is based?"
    )
    print(f"Test Case 1 Score: {score1}")  # 应该接近 100 (虽然泄露了 Lei Jun，扣点分，但逻辑对)

    # Case 2: 错误生成 (答案泄露)
    score2 = evaluate_question(
        triples_str=triples,
        start_node="Xiaomi Phone",
        end_node="Beijing",
        ref_question="What is the capital of the country where the inventor of Xiaomi Phone is based?",
        gen_question="Is Beijing the capital of China?"
    )
    print(f"Test Case 2 Score: {score2}")  # 应该很低