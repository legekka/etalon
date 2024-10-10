## Etalon Project
Etalon is a custom framework for Huggingface trainer libraries. It is designed to be as much unified between training different type of models as it could be, while still providing the required flexibility for the given model.

Mainly made for my own use, this documentation only written for my future selfs. But if you find it useful, you are welcome to use it.

## Branches!
- **main**: The main branch is the generalized trainer. It has some extra features, but it is focused on the framework itself. The changes in the main branch will be merged into the other branches.
- **T12**: The T12 branch is the TaggerNN-v12 trainer, which is a ViTForImageClassification model.
- **hu-DeBERTa-v2**: The hu-DeBERTa-v2 branch is the hu-DeBERTa-v2 Masked Language Modeling (MLM) trainer, which is a DeBERTa-v2 for Hungarian language.
- **aiops-categorizer**: The aiops-categorizer branch is a special class-token based classification model for ticket categorization.
- **ai-detector**: The ai-detector branch is a ViTForImageClassification model trainer for detecting AI generated images.