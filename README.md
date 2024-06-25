# kaggle_introduction_to_nlp

This repository is created for participating in [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) competition.

NLP, seq2seq tasks are listed below:

* one to many
* one to one
* many to one
* many to many

This project aims to predict which Tweets are about real disasters and which ones are not.

## Data

The [data](https://www.kaggle.com/competitions/nlp-getting-started/data) is available on the related kaggle competition page. Text vectorization and embedding are done on the dataset.

![data](images/data.png)

Here you can see the random data sampled from dataset and their labels:

![random_data](images/random_data.png)

## Run

You can run the notebook on Colab, Kaggle or jupyter notebook.

Clone the project

```bash
  git clone https://github.com/MrAliAmani/DisasterTweets
```

## Results

Different models have been tested on the dataset including a baseline model to compare other models with. You can see the sample predictions in the figure below:

![predictions](images/predictions.png)

* Model_0: Naive bayes (baseline model)

![baseline_model](images/baseline_model.png)

* Model_1: FFN

![model_1_fnn_loss_curves](images/model_1_fnn_loss_curves.png)

* Model_2: LSTM

![model_2_lstm_loss_curves](images/model_2_lstm_loss_curves.png)

* Model_3: GRU

![model_3_gru_loss_curves](images/model_3_gru_loss_curves.png)

* Model_4: BI-LSTM

![model_4_bilstm_loss_curves](images/model_4_bilstm_loss_curves.png)

* Model_5: Conv1D

![model_5_conv1d_loss_curves](images/model_5_conv1d_loss_curves.png)

* Model_6: Tensorflow_hub transfer learning (Universal Sentence Encoder, USE, as the feature extractor)

![model_6_use_loss_curves](images/model_6_use_loss_curves.png)

* Model_7: Transfer learning on 10% data

![model_7_use_10_loss_curves](images/model_7_use_10_loss_curves.png)

* Model_8: Sequential api FFN

* Model_9: LSTM sequential

* Model_10: Conv1D seq

* Model_11: Baseline retrain on 10% data

* model_12: Fine tuning tfhub USE model

* model_13: Ensemble model with LSTM model, Conv1D model and USE model.

![Ensemble model predictions](images/ensemble_preds.png)

### Model comparison

These models are compared based on accuracy, f1-score, precision and recall.

![model_comparison_table](images/model_comparison_table.png)

![model_comparison](images/model_comparison.png)

The models are sorted based on f1_score and the best model is identified.

![model_comparison_f1](images/model_comparison_f1.png)

Best model is model_6 (tfhub USE on all data). Its confusion matrix is shown below:

![confusion-matrix_best_model_validation_data](images/confusion-matrix_best_model_validation_data.png)

There has a tradeoff between speed and score of models. The baseline naive bayes model (model 11) shows better speed and the USE model (model 7) has better performance. Selection of the model depends on the use case and deployment. In the image below, model 7 and model 11 have been compared on 10 percent of data.

![compare-NB-USE-10](images/compare-NB-USE-10.png)

Finally the predictions are submitted to kaggle competition based on the specified format.

![submission_format](images/submission_format.png)

### Most wrong predictions

Identifying most wrong predictions can help us improve our model or detect some wrong labels in the dataset. Hers is the most wrong predictions that our model has done.

![most_wrong_predictions](images/most_wrong_predictions.png)

## Lessons Learned

* Using ensemble (stacking, blending, etc.) models can achieve good performance on textual data.
* There is a tradeoff between speed and score of different models that should be considered when deploying the models. Model 6 shows better performance (f1 score) but model 1 (baseline) is faster.

![model6_baseline](images/speed-score-tradeoff-model6_baseline.png)

## License

This project is under MIT license:

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

## Feedback

If you have any feedback, please reach out to me at *<aliamani019@gmail.com>*.

## Authors

[@AliAmani](https://github.com/MrAliAmani)
