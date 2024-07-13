# Project timeline
- idea development: May '24
- literature review: May-June '24
- model exploration, setup and training: June '24
- paper written: May-July '24

# Abstract
Predicting taxi demand is essential for managing urban transportation effectively. This study explores the application of next-frame prediction models—ConvLSTM and PredRNN—to forecast Origin-Destination (OD) taxi demand matrices using a concatenated dataset of NYC taxi data from early 2024. ConvLSTM achieved an RMSE of 1.27 with longer training times, while PredRNN achieved 1.59 with faster training. These models offer alternatives to traditional graph-based methods, showing strengths and trade-offs in real-world scenarios. Additionally, an open-source framework for model deployment is introduced, aiming to bridge the gap between research and practical implementation in taxi demand forecasting.

Keywords: Taxi, Demand, forecasting, OD Matrix, Next-Frame Prediction Models

# Technologies used
- **Terraform** to setup resources on GCP for MLflow
- **MLflow** used for experiment tracking
- **Prefect** used for pipeline orchestration

# Results
ConvLSTM and PredRNN were the main models used. Their trained weights (checkpoints) are saved in the `mlruns` folder.

| **Model**                | **RMSE** | **Train Time**  |
|--------------------------|----------|-----------------|
| Historical Average (HA)  | 21.54    | 42ms            |
| ConvLSTM                 | 1.27     | 3.2hr           |
| PredRNN                  | 1.59     | 2.2hr           |

# ConvLSTM prediction sample

![image](https://github.com/user-attachments/assets/7a488127-f73f-4ca1-989a-94179b46e221)

More can be found in the `sample_conv_lstm_pred_img` folder.

# Notes for the future
- explore more models
- use different data

# Notes on plagiarism
This paper is not published anywhere, I wrote it out of my own interest. If you found anything inside it useful, a ⭐ on the repo would be great. 

# Reproducability

If you would like to preproduce this project, you need to setup MLflow (either locally or on GCP/other cloud solution). You can follow the commands in [here](/setup_commands.md). Then run a Prefect flow with `python prefect_flows/file.py` from the root directory in the terminal.
