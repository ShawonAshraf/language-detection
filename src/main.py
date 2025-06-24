import mlcroissant as mlc
import pandas as pd
from fire import Fire
from loguru import logger
from sklearn.model_selection import train_test_split

from classifier import DocumentClassifier, NgramProfiler, calculate_distance
from evaluation import evaluate, infer

DATASET_URL = (
    "https://www.kaggle.com/datasets/basilb2s/language-detection/croissant/download"
)


def get_croissant_dataset(dataset_url: str = DATASET_URL) -> pd.DataFrame:
    logger.info(f"Fetching dataset from {dataset_url}")

    # Fetch the Croissant JSON-LD
    croissant_dataset = mlc.Dataset(dataset_url)

    # Check what record sets are in the dataset
    record_sets = croissant_dataset.metadata.record_sets

    # Fetch the records and put them in a DataFrame
    df = pd.DataFrame(croissant_dataset.records(record_set=record_sets[0].uuid))

    # Rename the columns
    df.rename(
        columns={
            "Language+Detection.csv/Text": "text",
            "Language+Detection.csv/Language": "language",
        },
        inplace=True,
    )

    logger.info(f"Dataset has {len(df)} rows")
    logger.info("Applying preprocessing steps to the dataset")

    # convert the binary strings to utf-8
    df["text"] = df["text"].apply(lambda x: x.decode("utf-8"))
    df["language"] = df["language"].apply(lambda x: x.decode("utf-8"))

    # correct spelling
    spelling = {
        "Sweedish": "Swedish",
        "Portugeese": "Portuguese",
    }
    df["language"] = df["language"].apply(lambda x: spelling.get(x, x))

    return df


def main(profile_size: int = 200) -> None:
    df = get_croissant_dataset()

    texts, languages = df["text"], df["language"]
    # train test split
    test_size = 0.2
    random_state = 42
    logger.info(
        f"Splitting dataset into train and test sets, {test_size} test size, {random_state} random state"
    )
    train_texts, test_texts, train_languages, test_languages = train_test_split(
        texts, languages, test_size=test_size, random_state=42
    )

    assert len(train_texts) == len(train_languages)
    assert len(test_texts) == len(test_languages)

    # train classifier
    classifier = DocumentClassifier(
        profiler=NgramProfiler(),
        profile_size=profile_size,
        distance_fn=calculate_distance,
    )
    classifier.fit(train_texts, train_languages)

    # evaluate classifier
    predictions = infer(classifier, test_texts)
    _, _, _, _, report = evaluate(predictions, test_languages)

    print(report)


if __name__ == "__main__":
    Fire(main)
