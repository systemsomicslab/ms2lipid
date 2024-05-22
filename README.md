# ms2lipid

ms2lipid is a Python library for predicted lipid class by using ms2 spectrum.

## Installation

Use the package manager pip to install ms2lipid.

```bash
pip install ms2lipid
```

## Usage
THe page URLs:
https://pypi.org/project/ms2lipid/0.1.3/

```python
import ms2lipid

# predicted lipid subclass by ms2 spectrum
predicted_result_df = ms2lipid.predclass(data_path, ms2spectrum_column_name, precurcerion_column_name, ionmode)

# Create a summary based on the information of correct class
ms2lipid.prediction_summary(predicted_result_df, data_path, class_correct_column_name)
```

## License

[MIT](https://choosealicense.com/licenses/mit/)
