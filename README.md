# PE-Fuse-Adapter: Perception Encoder Adaptation via Feature Interaction

This repository contains the implementation of `Fuse-Adapter`, a lightweight module for adapting, PE(Perception Encoder) to specific downstream tasks.

## Core Logic: The `Fuse_Adapter` Module

`Fuse-Adapter` learns to weigh and combine different forms of interaction between image and text features.

1.  **Feature Interaction**: For a given image and a set of class-describing texts, the adapter computes three types of interactions for each image-text pair:
    *   `concat = torch.cat([image_features, text_features])`
    *   `difference = image_features - text_features`
    *   `product = image_features * text_features`

2.  **Learnable Interaction Weights**: The model learns two sets of weights to control how these interactions are used:
    *   `interaction_weight` (shape: 3): A global set of weights for the three interaction types.
    *   `class_specific_adjustments` (shape: `num_classes`, 3): A per-class adjustment to the global weights.

    The final weights for each class are determined by `softmax(interaction_weight + class_specific_adjustments)`, allowing the model to decide, for instance, that the `product` interaction is more important for the 'fire' class, while the `difference` is more crucial for 'falldown'.

3.  **MLP Head**: The weighted interaction features are then processed by a simple Multi-Layer Perceptron (MLP) to produce a final logit for each class.

This approach allows the model to capture more complex relationships between visual and textual concepts than a standard classification head.

## Key Features

-   **Dynamic Feature Interaction**: Learns to combine concatenation, difference, and product of image-text embeddings on a per-class basis.
-   **Margin-based Training**: Optimizes the logit margins between target classes (`fire`, `falldown`) and a `normal` class, which is effective for imbalance and reducing false positives.
-   **Calibration Pipeline**: Includes a post-training calibration step that fits an affine transformation and finds a conservative decision threshold to improve model reliability.
-   **Checkpoint Management**: Automatically saves and manages the top-k checkpoints based on a validation selection score, tracking them in a `registry.json` file for each run.
-   **Hyperparameter Grid Search**: The entry point `main.py` is configured to run a full grid search over parameters defined in the config file.
-   **Comprehensive Video Benchmark Evaluation**: The `benchmark_eval.py` script provides a robust pipeline for evaluating trained models on video datasets, caching features and generating detailed performance reports.

## Further Details
For a more in-depth explanation of the model architecture, training methodology, and analysis of optimal hyperparameters, please refer to detailed documentation on Notion:

[PE-Fuse-Adapter: In-Depth Analysis & Documentation](https://jiminc.notion.site/HF-Test-24f0afc85c47804482d3f11eec45af16?source=copy_link)

## Project Structure

```
PE-Fuse-Adapter/
├── configs/
│   ├── image_benchmark_config.yaml
│   └── synthetic_caption_template_placeholder.json
├── core/
│   ├── model.py
│   └── trainer.py
├── data/
│   ├── test/
│   └── train/
├── logs/
│   ├── errors/
│   └── results/
├── model/
│   └── saved_models/
├── perception_models/  # External Git repository
├── preprocess/
│   ├── avg_text_features/
│   └── data_handler.py
├── benchmark_eval.py
├── main.py
└── utils.py
```

## Setup & Installation

#### 1. Clone Repository

```bash
git clone https://github.com/your-username/PE-Fuse-Adapter.git
cd PE-Fuse-Adapter
```

#### 2. Setup Perception Encoder

This project uses the [Perception Models from Facebook Research](https://github.com/facebookresearch/perception_models) as the vision encoder backbone. Clone it into the project root.

```bash
git clone https://github.com/facebookresearch/perception_models.git
```

**Note**: The code imports the encoder from `core.vision_encoder.pe`. If the structure of the cloned `perception_models` repo does not align, you may need to move its contents into `core/vision_encoder/` or create a symbolic link.

#### 3. Prepare Data

Place your training and testing images in the `data/` directory, following the `ImageFolder` structure.

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── test/
    ├── class_1/
    ├── class_2/
    └── ...
```

## How to Use

#### 1. Edit Configuration

Modify `configs/image_benchmark_config.yaml` to match your environment. Pay close attention to:

-   `paths.benchmark_dataset_root`: Set the absolute path to your benchmark video dataset.
-   `training.lr`, `interaction_lr_multiplier`, etc.: Define the hyperparameter ranges for the grid search.
-   `data.class_names`: Define the names of your classes.

#### 2. Run Training

Execute the following command to start the grid search and model training process.

```bash
python main.py
```

This will generate several outputs:
-   `model/saved_models/`: Stores the best model (`.pt`), calibration file (`_calib_ovn.json`), and a registry of top checkpoints (`_registry.json`) for each hyperparameter combination.
-   `logs/train.log`: A complete log of the training process.
-   `logs/training_log.csv`: A summary of each training run, including its hyperparameters and final validation score (`best_sel_score`).

#### 3. Run Benchmark Evaluation

After training, evaluate the models on your video benchmark dataset.

**To evaluate all checkpoints from all runs:**
```bash
python benchmark_eval.py
```

**To evaluate a specific model run:**
Use the `--tag` argument with a value from the `tag` column in `logs/training_log.csv`.
```bash
python benchmark_eval.py --tag "PE-Core-L14-336_lr0.0005_m4_wd0.001_d0.4_w10_b0.5"
```

**To evaluate a specific checkpoint rank from a model run:**
Use the `--rank` argument (from 1 to 5) along with `--tag`. This is useful for analyzing the performance of different checkpoints from the same training run.
```bash
python benchmark_eval.py --tag "..." --rank 1
```

This evaluation script produces:
-   `logs/results/{tag}/rank_{rank}/{class}/`: Frame-by-frame prediction CSVs for each video.
-   `logs/benchmark_evaluation_summary.csv`: A summary CSV with final metrics (Accuracy, F1-score, Precision, Recall) for every evaluated checkpoint.

## Understanding the Outputs

-   **`logs/training_log.csv`**: Your primary reference for finding the best-performing hyperparameter sets based on the validation `best_sel_score`.
-   **`logs/benchmark_evaluation_summary.csv`**: The final performance report. Use this to compare the generalization performance of different models on the benchmark dataset. The `avg_f1` column provides a high-level summary for ranking models.

## License

This project is distributed under the [MIT License](LICENSE).
