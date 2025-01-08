## WebSpotter Code

This repository hosts the source code and data for the WebSpotter paper.

In this work, we propose WebSpotter, a novel framework to provide the interpretability of deep learning-based web attack detection systems by locating malicious payloads within HTTP requests. WebSpotter segments HTTP requests into minimal semantic units (MSUs) and identifies which units contain malicious payloads. The method leverages both model behavior and textual semantics to provide interpretable and actionable insights into web attacks.

## Reproduction Steps

Below are the steps to reproduce the main results from the paper. The steps include training a web attack detection model, calculating importance scores of MSUs, training the payload localization model, and performing evaluations. The following example uses the CSIC dataset.

### Train the Detection Model

The first step is to train a detection model. A TextCNN model is used for this purpose, and the trained model will be saved in the tmp_model directory.

```
python classification/run.py --tmp_dir datasets/CSIC --tmp_model tmp_model --dataset csic --max_len 700
```

### Compute Importance Scores for MSUs

To localize malicious payloads, the importance scores for MSUs are calculated using the trained detection model. This step generates importance scores for both the training and testing datasets, and the results will be saved in the post_explain_result/csic/ folder.

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
## Note

Some sensitive data exists in our newly constructed datasets (FPAD, FPAD-OOD, and CVE attacks), which may pose a risk of de-anonymization for this repository. Therefore, we only showcase a few samples in these datasets. We will fully open-source these datasets once the review process is complete.

The complete CSIC and PKDD datasets, along with their location labels, are provided in this repository.
