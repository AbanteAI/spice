def fuzzy_model_lookup(model_hint):
    model_hint = str(model_hint).lower()

    models_list = [
        "gpt-4-0125-preview",
        "gpt-3.5-turbo-0125",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    for model in models_list:
        if model_hint in model:
            print(f"fuzzy matched: {model}")
            return model
