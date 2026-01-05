import json
from openai import OpenAI

base_url = 'https://openrouter.ai/api/v1'
model = 'openai/gpt-4o-mini'
api_key = ''


def generate_ref_question(path_str, start, end):
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
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
        model=model,
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


def generate_baseline_question(sample):
    """
    基于 Test 集中的子图样本生成问题 (用于 LLM Baseline)

    Args:
        sample (dict): 包含 'nodes', 'edge_index', 'edge_attr', 'label_ids' 的字典
                       (PyG Data 转换来的 dict 或者 HuggingFace Dataset 的 item)

    Returns:
        str: 生成的问题
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    # 1. 解析数据
    nodes = sample['nodes']
    edge_index = sample['edge_index']  # [[src...], [tgt...]]
    edge_attr = sample['edge_attr']  # ["rel1", "rel2"...]

    # 获取 Start 和 End 节点的文本
    # 注意: sample['label_ids'] 可能是 [start_idx, end_idx]
    # 在 build_dataset.py 里我们存的是 label_ids = [start_idx, end_idx]
    start_idx = sample['label_ids'][0]
    end_idx = sample['label_ids'][1]

    start_node_text = nodes[start_idx].replace("[TOPIC] ", "")
    end_node_text = nodes[end_idx].replace("[ANS] ", "")

    # 2. 构建三元组描述 (Triples Description)
    # 我们只保留正向关系 (不带 _inv 的)，以减少 Prompt 长度和干扰
    triples_desc = []

    # 处理 edge_index 格式 (可能是 PyG tensor 或者是 list)
    src_list = edge_index[0]
    tgt_list = edge_index[1]

    for i in range(len(src_list)):
        src = src_list[i]
        tgt = tgt_list[i]
        rel = edge_attr[i]

        # 过滤反向边 (通常包含 _inv)
        if "_inv" in rel:
            continue

        # 获取节点名
        # 注意: 列表索引
        if isinstance(src, int):
            # 如果是 list (HF Dataset)
            u = nodes[src].replace("[TOPIC] ", "").replace("[ANS] ", "")
            v = nodes[tgt].replace("[TOPIC] ", "").replace("[ANS] ", "")
        else:
            # 如果是 Tensor (PyG Data)
            u = nodes[src.item()].replace("[TOPIC] ", "").replace("[ANS] ", "")
            v = nodes[tgt.item()].replace("[TOPIC] ", "").replace("[ANS] ", "")

        triples_desc.append(f"({u}, {rel}, {v})")

    triples_str = "\n".join(triples_desc)

    # 3. Prompt 设计
    # 这里需要让 LLM 自己在图中找路径
    prompt = """
    You are an expert in Multi-Hop Question Generation over Knowledge Graphs.

    **Task:**
    I will provide a set of triples (Subject, Relation, Object) representing a Knowledge Graph Subgraph.
    Your goal is to generate a complex, multi-hop question where:
    1. The question is about the **Start Node**.
    2. The answer to the question is the **End Node**.
    3. The question implicitly follows a logical path connecting the Start Node to the End Node within the subgraph.

    **Constraints:**
    - The question must be natural and fluent.
    - **Do NOT** mention the name of the **End Node** (Answer) in the question. This is strictly forbidden.
    - **Do NOT** explicitly list the relations (e.g., don't say "linked by relation X"). Use natural language phrasing.
    - The question should require reasoning over the graph path (multi-hop).

    **Input Data:**
    - Subgraph Triples: A list of facts. Note that this subgraph may contain noise/irrelevant branches. You must find the path connecting Start to End.
    - Start Node: [Topic]
    - End Node: [Answer]

    **Output:**
    - Only output the generated question. Do not output any explanation.
    """

    user_content = f"""
    [Subgraph Triples]:
    {triples_str}

    [Start Node]: {start_node_text}
    [End Node]: {end_node_text}
    """

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0  # 保持评估稳定性
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM Baseline Error]: {e}")
        return ""


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
        base_url=base_url,
        api_key=api_key,
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
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},  # 强制 JSON 模式
            temperature=0.0  # 保持评估稳定性
        )

        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)

        score = int(result_json.get("score", 0))
        reason = str(result_json.get("reason", "No specific reason provided."))

        return score, reason

    except Exception as e:
        print(f"[LLM Eval Error]: {e}")
        # 如果出错，返回 -1 或者 0，视情况而定
        return -1, f"Error: {str(e)}"


if __name__ == "__main__":
    print(
        generate_ref_question(
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
    score1, _ = evaluate_question(
        triples_str=triples,
        start_node="Xiaomi Phone",
        end_node="Beijing",
        ref_question="What is the capital of the country where the inventor of Xiaomi Phone is based?",
        gen_question="Which city is the capital of the place where Lei Jun is based?"
    )
    print(f"Test Case 1 Score: {score1}")  # 应该接近 100 (虽然泄露了 Lei Jun，扣点分，但逻辑对)

    # Case 2: 错误生成 (答案泄露)
    score2, _ = evaluate_question(
        triples_str=triples,
        start_node="Xiaomi Phone",
        end_node="Beijing",
        ref_question="What is the capital of the country where the inventor of Xiaomi Phone is based?",
        gen_question="Is Beijing the capital of China?"
    )
    print(f"Test Case 2 Score: {score2}")  # 应该很低

    mock_sample = {
        "nodes": ["Roy Scheider", "multiple myeloma", "Susannah York", "[ANS] obesity", "leukemia", "[TOPIC] infection",
                  "old age"],
        "edge_index": [[1, 2, 1, 0, 1, 6, 1, 3, 3, 6, 5, 1, 5, 4], [2, 1, 0, 1, 6, 1, 3, 1, 6, 3, 1, 5, 4, 5]],
        "edge_attr": ["_people_cause_of_death_people", "_people_cause_of_death_people_inv",
                      "_people_cause_of_death_people", "_people_cause_of_death_people_inv",
                      "_medicine_disease_risk_factors", "_medicine_disease_risk_factors_inv",
                      "_medicine_disease_risk_factors", "_medicine_disease_risk_factors_inv",
                      "_medicine_disease_risk_factors", "_medicine_disease_risk_factors_inv",
                      "_medicine_symptom_symptom_of", "_medicine_symptom_symptom_of_inv",
                      "_medicine_symptom_symptom_of", "_medicine_symptom_symptom_of_inv"], "label_ids": [5, 3],
        "question": "What risk factors associated with the disease linked to an infection can contribute to obesity?"}
    q = generate_baseline_question(mock_sample)
    print(q)
