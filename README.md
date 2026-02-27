# LangLaw - Language Model Guided Symbolic Regression for Scientific Law Discovery


## Installation

### Quick Start with Docker (Recommended)

```bash
docker pull nakonako4/langlaw:latest
docker run --rm -it nakonako4/langlaw:latest python -c "import pysr; print('ready.')"
```

### Local Installation

```bash
# Clone the repository
cd /path/to/code

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Get an API Key

Get an API Key for Intern-S1: Register at [InternLM](https://internlm.intern-ai.org.cn/api/tokens) to obtain your API key.

You can also use other platform's API Key and URL to run LangLaw.

### 2. Configure Your Task

LangLaw uses YAML configuration files. Pre-configured examples are in `configs/`:

- `bulk_modulus.yaml` - Bulk modulus prediction for ABO₃ perovskites
- `band_gap.yaml` - Band gap prediction
- `oer.yaml` - OER activity prediction

**To use your own dataset:**

1. Create a new YAML config file (e.g., `configs/my_task.yaml`)
2. Specify your API Key, base url, data path, features, and target:

```yaml
llm:
  api_key: "sk-Ab12"  
  base_url: "https://chat.intern-ai.org.cn/api/v1/"
  model_name: "intern-s1"

data:
  data_path: "data/your_data.csv"
  experience_pool_path: "results/exp/exp_pool_your_task.json"
  results_dir: "results/your_task"
  target: "your_target_column"
  all_features:
    - feature1
    - feature2
    - feature3

pysr:
  niterations: 500
  maxdepth: 10

experiment:
  num_rounds: 20
  n_folds: 5
```

3. Create a prompt template in `prompts/your_task.txt` describing your scientific task

## Usage

### Run LLM-Guided Symbolic Regression

```bash
# Run a single fold
python scripts/main.py --config configs/bulk_modulus.yaml --fold 0

# Run all folds (5-fold cross-validation)
python scripts/main.py --config configs/bulk_modulus.yaml

# Use a different LLM model
python scripts/main.py --config configs/bulk_modulus.yaml --model gpt-4
```

### Run Baseline (No LLM Ablation)

```bash
# Baseline with K-fold
python scripts/baseline.py --config configs/bulk_modulus.yaml --fold 0

# Baseline on full dataset
python scripts/baseline.py --config configs/bulk_modulus.yaml --all-data
```

## Output

Results are organized in `results/your_task/<model>_<timestamp>_<fold>/`:

```
results/
└── bulk_modulus/
    └── intern-s1_2026-02-03_16-30-00_fold0/
        ├── fold_0_round_1_results.log    # Detailed results per round
        ├── fold_0_round_2_results.log
        ├── ...
        └── summary_all_folds.json        # Overall summary
```

Each result includes:
- **LLM reasoning** for feature selection
- **Selected features** and PySR parameters
- **Discovered formulas** (LaTeX and symbolic)
- **Performance metrics**: MAE, RMSE

## Structure

```
symlaw/
├── symlaw/                    # Main package
│   ├── config/               # Configuration management
│   ├── data/                 # Data loading utilities
│   ├── models/               # LLM selector & SR runner
│   └── utils/                # Logging and helpers
├── scripts/                  # Executable scripts
│   ├── main.py              # Main workflow
│   ├── baseline.py          # Baseline comparison
│   └── predict.py           # Prediction with trained models
├── configs/                  # YAML configuration files
├── prompts/                  # LLM prompt templates
├── data/                     # Data files
└── results/                  # Output results
```

## Custom Data Preprocessing

Create a custom preprocessor function:

```python
from symlaw.data.loader import five_fold_split

def my_preprocessor(X, feature_list):
    # Apply custom transformations
    X['new_feature'] = X['feat1'] * X['feat2']
    return X

# Use it
X_train, X_test, y_train, y_test = five_fold_split(
    data_path, features, target, fold_index=0,
    preprocessor=my_preprocessor
)
```

## Citation

If you use LangLaw in your research, please cite:

```bibtex
@misc{guan2026discoveryinterpretablephysicallaws,
      title={Discovery of Interpretable Physical Laws in Materials via Language-Model-Guided Symbolic Regression}, 
      author={Yifeng Guan and Chuyi Liu and Dongzhan Zhou and Lei Bai and Wan-jian Yin and Jingyuan Li and Mao Su},
      year={2026},
      eprint={2602.22967},
      archivePrefix={arXiv},
      primaryClass={physics.comp-ph},
      url={https://arxiv.org/abs/2602.22967}, 
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions or issues, please open an issue on GitHub.
