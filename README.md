## Language Detction from documents using n-gram profiles

This notebook is an attempt at building an n-gram profile based language detector inspired by [N-gram-based text categorization Cavnar, Trenkle (1994)](https://sdmines.sdsmt.edu/upload/directory/materials/12247_20070403135416.pdf).



#### BibTex entry
```bibtex
@inproceedings{Cavnar1994NgrambasedTC,
  title={N-gram-based text categorization},
  author={William B. Cavnar and John M. Trenkle},
  year={1994},
  url={https://api.semanticscholar.org/CorpusID:170740}
}
```

## Env Setup

Make sure to have `uv` installed before you proceed.

```bash
uv sync
source .venv/bin/activate
```

To run the example notebook,

```bash
jupyter notebook
```

Otherwise you can run the cli

```bash
uv run src/main.py PROFILE_SIZE
# example
uv run src/main.py 200
```
