## WebSpotter: Interpretable Web Attack Detection through Malicious Payload Localization

This repository hosts the source code and data for the paper "Achieving Interpretable Deep Learning-based Web Attack Detection through Malicious Payload Localization".

In this work, we propose WebSpotter, a novel framework to enhance the interpretability of deep learning-based web attack detection systems by locating malicious payloads within HTTP requests. WebSpotter segments HTTP requests into minimal semantic units (MSUs) and identifies which units contain malicious payloads. The method leverages both model behavior and textual semantics to provide interpretable and actionable insights into web attacks.

Due to anonymity requirements, only a small subset of FPAD, FPAD-OOD, and Pocrest datasets is made publicly available in this repository for demonstration purposes.

## Reproduction Steps

Below are the steps to reproduce the main results from the paper. The steps include training a web attack detection model, calculating importance scores of MSUs, training the payload localization model, and performing evaluations. The following example uses the CSIC dataset.

### Train a TextCNN Classification Model

The first step is to train a classification model to detect whether HTTP requests are benign or malicious. A TextCNN model is used for this purpose, and the trained model will be saved in the tmp_model directory.

```
python classification/run.py --tmp_dir datasets/CSIC --tmp_model tmp_model --dataset csic --token char --max_len 700
```

### Compute Importance Scores for MSUs

To localize malicious payloads, the importance scores for MSUs are calculated using the trained classification model. This step generates importance scores for both the training and testing datasets, and the results will be saved in the post_explain_result/csic/ folder.

```
python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-csic-512-None-0.pth \
    --outputdir post_explain_result/csic/test \
    --dataset csic \
    --test_path datasets/CSIC/test.jsonl

python localization/post_explain/run_explain.py \
    --model_path tmp_model/textcnn-700-csic-512-None-0.pth \
    --outputdir post_explain_result/csic/train \
    --dataset csic \
    --test_path datasets/CSIC/train.jsonl
```

### Train the Localization Model and Evaluate

This step trains the localization model to identify malicious MSUs and evaluates its performance. 

```
python localization/binary_based/run.py \
    --feature_method score_sort_with_textemb \
    --dataset csic \
    --train_path post_explain_result/csic/train/train.jsonl_withscore \
    --test_path post_explain_result/csic/test/test.jsonl_withscore \
    --output_path binary_result/csic \
    --sample_rate 0.01
```