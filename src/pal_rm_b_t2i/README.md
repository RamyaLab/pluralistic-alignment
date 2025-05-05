
for the text-2-image (T2I) model, we do the following modifications:

1. `itemLearner` class `forward()` method is modified to omit the `attention_mask` inputs
2. We only apply the `tiny-version` of our PAL models on T2I datasets: we use the fixed base model to generate the image embeddings, thus we omit the base_model inference in the `forward()` method. 
3. 