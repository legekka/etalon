# Etalon Project
Etalon is a custom framework for Huggingface trainer libraries. It is designed to be as much unified between training different type of models as it could be, while still providing the required flexibility for the given model.

Mainly made for my own use, this documentation only written for my future selfs. But if you find it useful, you are welcome to use it.

## Branches!
- `main`: This is the generalized trainer. It has some extra features, but it is focused on the framework itself. The changes in the main branch will be merged into the other branches.

- `T12`: This is the TaggerNN-v12 trainer, which is a **ViTForImageClassification** model.

- `hu-DeBERTa-v2`: This is the hu-DeBERTa-v2 Masked Language Modeling (MLM) trainer, which is a **DeBERTa-v2** for Hungarian language.

- `aiops-categorizer`: This branch is a special class-token based classification model for ticket categorization.

- `ai-detector`: This branch is a **ViTForImageClassification** model trainer for detecting AI generated images.