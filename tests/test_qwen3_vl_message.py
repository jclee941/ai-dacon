from skku_vqa.models.qwen3_vl import build_message_content


def test_image_item_includes_max_pixels_when_set() -> None:
    content = build_message_content("/img/x.jpg", "pick one", max_pixels=100352)

    image_items = [c for c in content if c.get("type") == "image"]
    assert len(image_items) == 1
    assert image_items[0]["image"] == "/img/x.jpg"
    assert image_items[0]["max_pixels"] == 100352
    assert image_items[0]["min_pixels"] == 28 * 28 * 4


def test_text_item_always_present_and_no_image_when_none() -> None:
    content = build_message_content(None, "pick one", max_pixels=100352)

    assert all(c.get("type") != "image" for c in content)
    assert content[-1] == {"type": "text", "text": "pick one"}


def test_image_item_omits_max_pixels_when_unset() -> None:
    content = build_message_content("/img/x.jpg", "pick one", max_pixels=None)

    image_items = [c for c in content if c.get("type") == "image"]
    assert "max_pixels" not in image_items[0]
