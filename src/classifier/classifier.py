from typing import Callable

from loguru import logger
from tqdm.auto import tqdm

from .profiler import NgramProfiler


class DocumentClassifier:
    def __init__(self, profiler: NgramProfiler, profile_size: int, distance_fn: Callable):
        logger.info(f"Initializing DocumentClassifier with profile size {profile_size}")
        self.profiler = profiler
        self.profile_size = profile_size
        self.distance_fn = distance_fn

        self.profiles = {}

    def fit(self, docs: list[str], languages: list[str]):
        logger.info("Fitting classifier")
        logger.info(f"Languages: {len(languages)}")
        logger.info(f"Docs: {len(docs)}")
        
        data_dict = {lang: [] for lang in languages}
        # collect all texts for one language
        for doc, language in tqdm(zip(docs, languages), total=len(docs)):
            if language not in data_dict:
                data_dict[language] = [doc]
            else:
                data_dict[language].append(doc)

        # generate profiles
        logger.info("Generating profiles")
        for category in data_dict.keys():
            text = " ".join(data_dict[category])
            profile = self.profiler.generate_profile(text)
            self.profiles[category] = profile[:self.profile_size]

    def predict(self, doc: str):
        doc_profile = self.profiler.generate_profile(doc)

        category_distances = {}
        for category in self.profiles.keys():
            distance = self.distance_fn(doc_profile, self.profiles[category])
            category_distances[category] = distance

        return min(category_distances, key=category_distances.get)
