[![](https://img.shields.io/pypi/v/ms2lipid.svg?maxAge=3600)](https://pypi.org/project/ms2lipid/)

# ms2lipid

ms2lipid is a a lipid subclass prediction program using machine learning and curated tandem mass spectral data.

## Installation

```bash
pip install ms2lipid
```

## Quick start

```python
import ms2lipid

# Lipid subclass prediction by ms2 spectrum
# data_path is the file path of csv, msp, or txt
predicted_result_df = ms2lipid.predclass(data_path, ms2spectrum_column_name, precurcerion_column_name, ionmode)

# Create a summary based on the information of correct class
ms2lipid.prediction_summary(predicted_result_df, data_path, class_correct_column_name)
```

## Documentation
https://systemsomicslab.github.io/ms2lipid/
