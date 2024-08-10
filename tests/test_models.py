import pytest
from spice.models import get_model_from_name, OPEN_AI

def test_gpt_4_0806_model():
    model = get_model_from_name("gpt-4-0806")
    assert model.name == "gpt-4-0806"
    assert model.provider == OPEN_AI
    assert model.input_cost == 300  # Adjust to the correct input cost
    assert model.output_cost == 600  # Adjust to the correct output cost
    assert model.context_length == 128000  # Adjust if the context length is different