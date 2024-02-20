# ms2lipid

ms2lipid is a Python library for predicted lipid class by using ms2 spectrum.

## Installation

Use the package manager pip to install ms2lipid.

```bash
pip install -i https://test.pypi.org/simple/ ms2lipi
pip install ms2lipid #This is not yet implemented.
```

## Usage

```python
import ms2lipid

# predicted lipid subclass by ms2 spectrum
predicted_result_df = ms2lipid.from_data_pred_classs(ms2spectrum(series), precursorion(series), ionmode)

# Create a summary based on the information of correct class
ms2lipid.prediction_summary(predicted_result_df, correctclass(series))

# save predicted file as csv
ms2lipid.save_pred_result(predicted_result_df, 'export_result_df.csv')
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
