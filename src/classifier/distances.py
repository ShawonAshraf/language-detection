def calculate_distance(doc_profile: list[str], category_profile: list[str], no_match_penalty: int | None = None) -> int:
    """
    Calculates the 'out-of-place' distance between two N-gram profiles.

    This measure is based on the description in "N-Gram-Based Text Categorization".
    It computes how far out of place N-grams in the document profile are
    compared to their positions in the category profile.

    Args:
        doc_profile: A list of N-grams for the document, sorted by frequency.
        category_profile: A list of N-grams for the category, sorted by frequency.
        no_match_penalty: The penalty to apply for an N-gram in the document
                          profile that is not found in the category profile.
                          If None, it defaults to the length of the category profile.

    Returns:
        The total out-of-place distance between the two profiles.
    """
    # Create a map of N-gram to rank for the category profile for efficient lookup
    category_ranks = {ngram: i for i, ngram in enumerate(category_profile)}

    # Set the penalty for N-grams not found in the category profile.
    # The paper suggests a "maximum out-of-place value". The length of the
    # profile is a reasonable choice for this maximum penalty.
    if no_match_penalty is None:
        no_match_penalty = len(category_profile)

    total_distance = 0

    # Iterate through the document profile to calculate the distance
    for doc_rank, ngram in enumerate(doc_profile):
        if ngram in category_ranks:
            category_rank = category_ranks[ngram]
            # Calculate how far out of place the N-gram is
            distance = abs(doc_rank - category_rank)
        else:
            # Apply penalty if the N-gram is not in the category profile
            distance = no_match_penalty
        total_distance += distance

    return total_distance
