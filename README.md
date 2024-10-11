# ms2lipid

ms2lipid is a Python library for predicted lipid class by using ms2 spectrum.

## Installation

Use the package manager pip to install ms2lipid.

```bash
pip install ms2lipid
```

## Usage

```python
import ms2lipid

# predicted lipid subclass by ms2 spectrum
predicted_result_df = ms2lipid.predclass(data_path, ms2spectrum_column_name, precurcerion_column_name, ionmode)

```

## License

[MIT](https://choosealicense.com/licenses/mit/)
