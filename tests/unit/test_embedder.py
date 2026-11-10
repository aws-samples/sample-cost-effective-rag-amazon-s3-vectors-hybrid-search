import json

from ingestion.embedder import embed_text, embed_texts


def test_embed_text_request_contract(fake_bedrock):
    vector = embed_text("hello world", dimensions=8)

    assert len(vector) == 8
    assert all(isinstance(v, float) for v in vector)

    call = fake_bedrock.invoke_calls[-1]
    assert call["modelId"] == "amazon.titan-embed-text-v2:0"
    body = json.loads(call["body"])
    assert body == {"inputText": "hello world", "dimensions": 8, "normalize": True}


def test_embed_text_uses_configured_dimensions(fake_bedrock):
    embed_text("hello")  # EMBEDDING_DIMENSIONS=8 from conftest env
    body = json.loads(fake_bedrock.invoke_calls[-1]["body"])
    assert body["dimensions"] == 8


def test_embed_text_truncates_to_titan_char_limit(fake_bedrock):
    embed_text("x" * 60000)
    body = json.loads(fake_bedrock.invoke_calls[-1]["body"])
    assert len(body["inputText"]) == 50000


def test_embed_texts_one_call_per_text(fake_bedrock):
    vectors = embed_texts(["one", "two", "three"])
    assert len(vectors) == 3
    assert len(fake_bedrock.invoke_calls) == 3
