import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    import re
    import hashlib

    text = example["text"]

    # Make the transformation reproducible for each review.
    seed = int.from_bytes(
        hashlib.sha256(text.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    rng = random.Random(seed)

    typo_probability = 0.45

    # Preserve words that directly express negation.
    protected = {
        "no", "not", "never", "neither", "nor",
        "none", "nobody", "nothing", "nowhere",
        "without", "hardly", "barely", "scarcely",
        "cannot",
    }

    replacements = []

    # Match contractions as whole tokens so they can be skipped.
    pattern = r"[A-Za-z]+(?:['’][A-Za-z]+)*"

    for match in re.finditer(pattern, text):
        word = match.group()

        if not word.isalpha() or len(word) < 5:
            continue

        if word.lower() in protected:
            continue

        # Preserve capitalized words to reduce changes to names.
        if not word.islower():
            continue

        # Swap only internal letters, preserving first and last.
        positions = [
            i for i in range(1, len(word) - 2)
            if word[i] != word[i + 1]
        ]

        if not positions or rng.random() >= typo_probability:
            continue

        i = rng.choice(positions)
        letters = list(word)
        letters[i], letters[i + 1] = letters[i + 1], letters[i]

        replacements.append(
            (match.start(), match.end(), "".join(letters))
        )

    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]

    example["text"] = text
    ##### YOUR CODE ENDS HERE ######

    return example
