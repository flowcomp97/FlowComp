import _bootstrap  # noqa: F401

from datasets import Dataset, load_dataset
import transformers

from flowcomp import DataArguments, FlowComp, ModelArguments, TrainingArguments
from flowcomp.training_utils import FlowCompDataCollator, tokenize_function, train_model


def preprocess_split(split, model):
    records = []
    for example in split:
        tokenized = tokenize_function({key: [value] for key, value in example.items()}, model)
        records.append({key: value[0] for key, value in tokenized.items()})
    return Dataset.from_list(records)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model = FlowComp(model_args, training_args)

    if not data_args.train_file:
        raise ValueError("--train_file is required")

    data_files = {"train": data_args.train_file}
    if data_args.eval_file:
        data_files["eval"] = data_args.eval_file

    dataset = load_dataset("json", data_files=data_files, streaming=data_args.streaming)
    if data_args.streaming:
        train_dataset = dataset["train"].map(
            tokenize_function,
            batched=True,
            fn_kwargs={"model": model},
        )
        eval_dataset = None
        if "eval" in dataset:
            eval_dataset = dataset["eval"].map(
                tokenize_function,
                batched=True,
                fn_kwargs={"model": model},
            )
    else:
        train_dataset = preprocess_split(dataset["train"], model)
        eval_dataset = preprocess_split(dataset["eval"], model) if "eval" in dataset else None

    data_collator = FlowCompDataCollator(
        source_pad_token_id=model.source_pad_token_id,
        target_pad_token_id=model.target_pad_token_id,
    )
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)


if __name__ == "__main__":
    main()
