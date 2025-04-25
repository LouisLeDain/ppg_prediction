# Phoneme prediction

**Current idea** : 
- Build up a pipeline consisting of a feature extractor with pre-trained [Wav2Vec 960H model](https://arxiv.org/pdf/2006.11477v3) then followed by a transformer-based PPG generator (handmade ?)
- Maybe : Fine-tune on `L2-arctic`dataset (discussion on metrics and loss functions...)