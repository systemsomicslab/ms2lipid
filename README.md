[![](https://img.shields.io/pypi/v/ms2lipid.svg?maxAge=3600)](https://pypi.org/project/ms2lipid/)
[![DOI:10.1101/2024.05.16.594510](http://img.shields.io/badge/DOI-10.1101/2024.05.16.594510-B31B1B.svg)](https://doi.org/10.1101/2024.05.16.594510)

# ms2lipid
MS2Lipid is a a lipid subclass prediction program using machine learning and curated tandem mass spectral data.

## Installation
`ms2lipid` PyPI package is available for Python **3.9 to 3.12**.

```bash
pip install ms2lipid
```

## Try MS2Lipid on Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/systemsomicslab/ms2lipid/blob/main/try_ms2lipid.ipynb)

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Building `ms2lipid` Python package
Make sure you have the latest version of PyPA’s build installed:
```
python -m pip install --upgrade build
# or
# python3 -m pip install --upgrade build
```

Now run this command from the same directory where pyproject.toml is located:
```
python -m build
# or
# python3 -m build
```
