from classifier import DocumentClassifier, NgramProfiler, calculate_distance


if __name__ == "__main__":
    classifier = DocumentClassifier(
        profiler=NgramProfiler(),
        profile_size=200,
        distance_fn=calculate_distance
    )
