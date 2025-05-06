from fi_instrumentation.fi_types import EvalName, EvalTag, EvalTagType, EvalSpanKind

eval_tags = [
    EvalTag(
        eval_name=EvalName.DETERMINISTIC_EVALS,
        type=EvalTagType.OBSERVATION_SPAN,
        value=EvalSpanKind.AGENT,
        config={
            "rule_prompt": """
            Check if the formula is accurate and is what the user asked for based on the user's query 

            query: ({{raw.input}})

            formula: ({{raw.output}})
            """,
            "choices": ["Yes", "No"],
            "multi_choice": False
        },
        mapping={},
        custom_eval_name="Formula Accuracy"
    ),
    EvalTag(
        eval_name=EvalName.DETERMINISTIC_EVALS,
        type=EvalTagType.OBSERVATION_SPAN,
        value=EvalSpanKind.LLM,
        config={
            "rule_prompt": """
            Check if the table name extracted from the user's query is correct

            query: ({{llm.input_messages.1.message.content}})

            table_name: ({{table_name}})
            """,
            "choices": ["Yes", "No"],
            "multi_choice": False
        },
        mapping={},
        custom_eval_name="Table Selection"
    )
]
