text_path = Path(
        r"C:\Users\rosie\PycharmProjects\AIBigProject\data\raw\tiny_shakespeare.txt"
    )

    text = text_path.read_text(encoding="utf-8")

    split = int(len(text) * 0.9)

    split = text.find("\n", split)

    train_text = text[:split]
    eval_text = text[split:]

    Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\train').write_text(train_text, encoding="utf-8")
    Path(r'C:\Users\rosie\PycharmProjects\AIBigProject\data\processed\tiny_shake\eval').write_text(eval_text, encoding="utf-8")