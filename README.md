# How Does a Machine Learning Model Learn from Data?

A practical MovieLens experiment demonstrating how a model learns from rating errors, how validation guides model selection, and how held-out test data measures generalization.

The matrix factorization model reduced test MAE by **29.56%** and test RMSE by **25.87%** compared with predicting the training mean for every user–movie pair.

## Prediction Task

Given a user and a movie, predict the rating that user might give. This is supervised regression using observed user–movie ratings as targets.

| Field | Role |
|---|---|
| `userId` | User identifier |
| `movieId` | Movie identifier |
| `rating` | Numerical prediction target |
| `timestamp` | Retained metadata; unused in the random split |

IDs are categorical identifiers, not ordered numerical features. Titles and genres support exploration but are not inputs to the initial model.

## Dataset

**MovieLens `ml-latest`, generated July 20, 2023**, provided by GroupLens Research.

| Property | Value |
|---|---:|
| Ratings | 33,832,162 |
| Users | 330,975 |
| Movies in metadata | 86,537 |
| Rating scale | 0.5–5.0 stars, in half-star increments |

The notebook initially uses `ratings.csv` and `movies.csv`. Additional files include `tags.csv`, `links.csv`, `genome-scores.csv`, and `genome-tags.csv`.

Download from [GroupLens](https://grouplens.org/datasets/movielens/latest/). `ml-latest` is a development dataset that can change; retain the downloaded README and release information when reproducing this experiment. A newer download may produce different results.

## Notebook Workflow

1. Define the prediction problem.
2. Introduce the dataset and release.
3. Load and inspect the files.
4. Check missing values, duplicate pairs, rating values, and metadata matches.
5. Explore rating distributions and user/movie rating counts.
6. Define inputs and target.
7. Randomly split ratings into training, validation, and test sets.
8. Establish a training-mean baseline.
9. Encode training IDs and train matrix factorization.
10. Inspect learning curves and select the best validation epoch.
11. Evaluate the selected model on held-out test ratings.
12. Explain findings and limitations.

## Model

The prediction combines a fixed training mean, learned user and movie biases, and the dot product of learned vectors:

$$\hat r_{u,m}=\mu+b_u+b_m+\mathbf p_u^\top\mathbf q_m$$

Each user and movie vector contains **32 trainable values**. Vector dimensions are learned from interactions and are not assigned genre meanings.

| Setting | Value |
|---|---|
| Framework | TensorFlow/Keras |
| Embedding dimension | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training loss | Mean squared error |
| Batch size | 8,192 |
| Maximum epochs | 10 |
| Early stopping | Validation RMSE, patience 2, restore best weights |
| Random seed | 42 |
| Split | 80% training / 10% validation / 10% test |

ID mappings are built from training data only. If either identifier is unseen, the prediction falls back to the training mean. Evaluation uses raw predictions without clipping.

## Results

| Test metric | Training-mean baseline | Model + fallback | Error reduction |
|---|---:|---:|---:|
| MAE | 0.8427 stars | **0.5936 stars** | **29.56%** |
| RMSE | 1.0639 stars | **0.7886 stars** | **25.87%** |

**Selected epoch: 3.** Training stopped after epoch 5, with epoch-3 weights restored. Best known-ID validation RMSE was **0.7886**. Test ratings requiring fallback accounted for **0.10%** of the test set.

MAE measures average absolute error in stars. RMSE gives greater weight to larger errors. Error-reduction percentages are not accuracy percentages. Both final models were evaluated on the same test rows.

### Learning Progress

| Epoch | Training RMSE | Known-ID validation RMSE |
|---|---:|---:|
| 1 | 0.9073 | 0.8308 |
| 2 | 0.7932 | 0.7976 |
| **3** | **0.7447** | **0.7886** |
| 4 | 0.7121 | 0.7890 |
| 5 | 0.6899 | 0.7928 |

After epoch 3, training error continued decreasing while validation error increased slightly, suggesting the beginning of overfitting. Validation guided epoch selection; test ratings were reserved for final evaluation.

## Visualizations

The notebook includes:

- Rating counts and percentages by star value.
- Ratings per user and per movie, using logarithmic horizontal axes.
- Average movie rating versus rating count.
- Training and validation MAE/RMSE learning curves.
- Baseline versus learned-model validation and test comparisons.

<!-- Optional: upload movielens_test_performance.png to the repository root,
then uncomment the following image to display it.
![Final test MAE and RMSE comparison](movielens_test_performance.png)
-->

## Run Locally

Install the notebook dependencies in a compatible Python environment:

```bash
python -m pip install jupyter numpy pandas matplotlib seaborn tensorflow
```

1. Download and extract the dataset separately; keep large CSVs out of Git.
2. Open the project notebook in Jupyter.
3. Update `DATA_DIR` to the extracted folder containing `ratings.csv` and `movies.csv`.
4. Run cells in order, starting with imports and visualization settings.
5. Resolve any reported data-quality issues before splitting or training.

```python
from pathlib import Path

DATA_DIR = Path(r"C:\path\to\ml-latest")
```

```bash
jupyter notebook
```

The full dataset contains more than 33 million ratings. Loading, splitting, encoding, and TensorFlow tensors require additional memory beyond the raw table. Full epochs can take substantial time, depending on hardware. Seeds improve reproducibility, but framework versions and hardware can affect results.

<!-- Add the actual notebook filename/link here after uploading it.
Add an article permalink here once the article has been published.
-->

## Limitations

- A random split measures held-out ratings across the dataset period, not future prediction performance.
- Most test users and movies appeared in training; cold-start performance is not established by these aggregate results.
- Users choose what to watch and rate, creating selection effects in observed data.
- Lightly rated users and movies provide limited evidence.
- Raw predictions can fall outside the 0.5–5.0 scale.
- Rating errors do not measure top-N recommendation quality.
- Improvement over a constant baseline does not establish superiority over other recommender models.

## Data Usage and License

MovieLens is governed by its own usage conditions; public availability does not imply unrestricted use. Its README requires acknowledgment and permission for commercial or revenue-bearing use. Review the [official dataset terms](https://files.grouplens.org/datasets/movielens/ml-latest-README.html) before using or redistributing the data.

No software license is declared in this README. Add a separate `LICENSE` file if a code license is selected.

## References

- Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems* (3rd ed.). O’Reilly Media. [Publisher](https://www.oreilly.com/library/view/hands-on-machine-learning/9781098125967/)
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press. [Open-access book](https://www.deeplearningbook.org/)
- GroupLens Research. (2023). *MovieLens latest datasets* [Data set; July 20, 2023 release]. [Documentation](https://files.grouplens.org/datasets/movielens/ml-latest-README.html)
- Harper, F. M., & Konstan, J. A. (2015). The MovieLens datasets: History and context. *ACM Transactions on Interactive Intelligent Systems, 5*(4), Article 19. https://doi.org/10.1145/2827872
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer, 42*(8), 30–37. https://doi.org/10.1109/MC.2009.263

## Author

Wardatul Keskin — [BrightMind Engineer](https://brightmindengineer.blogspot.com/)
