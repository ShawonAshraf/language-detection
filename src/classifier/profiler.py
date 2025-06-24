import re
from collections import Counter


class NgramProfiler:
    """
    An N-gram profiler based on the method described in the paper
    "N-Gram-Based Text Categorization" by Cavnar and Trenkle.

    This class generates a frequency profile of N-grams for a given text.
    The profile is a list of N-grams sorted in descending order of their
    occurrence frequency.

    The process involves:
    1. Tokenizing the text, keeping only alphabetic characters and apostrophes.
    2. Generating N-grams of lengths 1 to 5 for each token.
    3. Counting the frequency of each N-gram.
    4. Sorting the N-grams by frequency to create the final profile.
    """

    def __init__(self, n_min=1, n_max=5):
        """
        Initializes the NgramProfiler.

        Args:
            n_min (int): The minimum length of N-grams to generate.
            n_max (int): The maximum length of N-grams to generate.
        """
        self.n_min = n_min
        self.n_max = n_max

    def _tokenize(self, text: str) -> list[str]:
        """
        Splits text into tokens, keeping only letters and apostrophes.
        Digits and punctuation are discarded, and text is lowercased. 

        Args:
            text: The input string.

        Returns:
            A list of cleaned and lowercased tokens.
        """
        # Keep only letters and apostrophes, discard others
        text = re.sub(r'[^a-zA-Z\']+', ' ', text)
        return text.lower().split()

    def _generate_ngrams_for_token(self, token: str) -> list[str]:
        """
        Generates all possible N-grams for a single token for N=1 to 5. 
        The token is padded with spaces to handle beginning and end of words. 

        Args:
            token: A single word string.

        Returns:
            A list of all N-grams for the token.
        """
        # Pad the token with a single space on each side
        padded_token = f" {token} "
        ngrams = []
        # Scan down the token to generate N-grams from n_min to n_max
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(padded_token) - n + 1):
                ngrams.append(padded_token[i:i+n])
        return ngrams

    def generate_profile(self, text: str) -> list[str]:
        """
        Generates an N-gram frequency profile for the given text.

        The process follows the description in the paper:
        - Tokenizes the text. 
        - Generates N-grams for each token. 
        - Counts the occurrences of all N-grams. 
        - Sorts the N-grams by frequency in reverse order. 
        - Returns the list of N-grams, which constitutes the profile. 

        Args:
            text: The input document as a string.

        Returns:
            A list of N-grams sorted by frequency in descending order.
        """
        tokens = self._tokenize(text)
        all_ngrams = []
        for token in tokens:
            all_ngrams.extend(self._generate_ngrams_for_token(token))

        # Count the occurrences of all N-grams
        ngram_counts = Counter(all_ngrams)

        # Sort N-grams into reverse order by the number of occurrences
        sorted_ngrams = sorted(ngram_counts.keys(),
                               key=lambda x: ngram_counts[x], reverse=True)

        return sorted_ngrams
