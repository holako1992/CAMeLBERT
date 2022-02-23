# CAMeLBERT: A collection of pre-trained models for Arabic NLP tasks:

This repo contains code for the experiments presented in our paper: [The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models](https://arxiv.org/pdf/2103.06678.pdf).

## Requirements:

This code was written for python>=3.7, pytorch 1.5.1, and transformers 3.1.0. You will also need few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/CAMeL-Lab/CAMeLBERT.git
cd CAMeLBERT

conda create -n CAMeLBERT python=3.7
conda activate CAMeLBERT

pip install -r requirements.txt
```

## Token Classification:

### NER:

For the NER experiments, we used the [ANERCorp](https://link.springer.com/chapter/10.1007/978-3-540-70939-8_13) dataset and followed the splits defined by [Obeid et al., 2020](https://camel.abudhabi.nyu.edu/anercorp/).
The dataset doesn't have a dev split, so we fine-tune the models on the train split and evaluate the last checkpoint on the test split.
To run the fine-tuning:


```bash
export DATA_DIR=/path/to/data                 # Should contain train/dev/test/labels files
export MAX_LENGTH=512
export BERT_MODEL=/path/to/pretrained_model/  # Or huggingface model id
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=12345

 python run_token_classification.py \
  --data_dir $DATA_DIR \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_train \
  --do_predict
```


## Citation:

If you find any of the CAMeLBERT or the fine-tuned models useful in your work, please cite [our paper](https://arxiv.org/pdf/2103.06678.pdf):
```bibtex
@inproceedings{inoue-etal-2021-interplay,
    title = "The Interplay of Variant, Size, and Task Type in {A}rabic Pre-trained Language Models",
    author = "Inoue, Go  and
      Alhafni, Bashar  and
      Baimukan, Nurpeiis  and
      Bouamor, Houda  and
      Habash, Nizar",
    booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
    month = apr,
    year = "2021",
    address = "Kyiv, Ukraine (Online)",
    publisher = "Association for Computational Linguistics",
    abstract = "In this paper, we explore the effects of language variants, data sizes, and fine-tuning task types in Arabic pre-trained language models. To do so, we build three pre-trained language models across three variants of Arabic: Modern Standard Arabic (MSA), dialectal Arabic, and classical Arabic, in addition to a fourth language model which is pre-trained on a mix of the three. We also examine the importance of pre-training data size by building additional models that are pre-trained on a scaled-down set of the MSA variant. We compare our different models to each other, as well as to eight publicly available models by fine-tuning them on five NLP tasks spanning 12 datasets. Our results suggest that the variant proximity of pre-training data to fine-tuning data is more important than the pre-training data size. We exploit this insight in defining an optimized system selection model for the studied tasks.",
}
```
