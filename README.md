# Phoneme prediction

**Current idea** : 
- Use [VAE Based Phoneme Alignment](https://arxiv.org/html/2407.02749v1) to obtain ground-truth ppgs of the `basic5000` dataset.
- Build up a pipeline consisting of a feature extractor with pre-trained [Japanese Speech Recognition model](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-japanese) then followed by a transformer-based PPG generator (handmade ?)
- Train on `basic5000`dataset (discussion on metrics and loss functions...)