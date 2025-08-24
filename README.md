# Inter-IIT-PS-24  
**GIF Feature Extraction and Description Generation**  

### Contributors:
- **Anshuman Nivas** (EEE’26)
- **Manoj Sunuguri** (EEE’26)

---

### Introduction  
This project focuses on associating GIF data with textual descriptions by leveraging a pre-trained ResNet50 model for GIF feature extraction and an RNN-based architecture to model the relationship between the visual features and the corresponding text descriptions. The goal is to build a model that can predict the textual description of a GIF based on its extracted visual features.

---

### Approach  

#### 1. Feature Extraction using ResNet50  

- **Frame Extraction**  
  Each GIF consists of multiple frames, but not all are equally important. A method was developed to extract 16 key frames from each GIF, using a sampling strategy that calculates step size based on the total number of frames. The frames were resized to 224x224 pixels and normalized for ResNet50.

- **Feature Extraction**  
  The pre-trained ResNet50 model, with its fully connected layer removed, was used to extract a 2048-dimensional feature vector for each frame.

- **Positional Encoding**  
  Since GIFs are sequences of frames, positional encoding (using sine and cosine functions) was added to capture the order of the frames.
---

#### 2. Dataset Preparation  

- **Loading GIF Features**  
  The output from ResNet model were passed containing a matrix of ResNet features for a specific GIF.

- **Loading Descriptions**  
  Text descriptions for each GIF were stored in text files. These descriptions were loaded for training purposes.

- **Tokenization**  
  The BERT tokenizer (`BertTokenizer`) was used to tokenize the descriptions, padding/truncating them to a length of 128 tokens. These tokens were paired with their respective GIF features.

---

#### 3. Model Architecture  

- **RNN for GIF Features**  
  The ResNet-extracted features were passed through an RNN, which learned the temporal dependencies between frames.

- **Text Embedding**  
  The tokenized descriptions were transformed into dense vector representations using a randomly initialized embedding layer.

- **Fusion of GIF and Text Features**  
  The outputs from the RNN (GIF features) and the text embedding were combined using element-wise addition. This simple combination can be improved with more advanced fusion techniques in the future.

- **Output**  
  The final model output was a sequence of predicted tokens, reconstructing the GIF's textual description. The model was trained to minimize the difference between the predicted and actual tokens.

---

#### 4. Training and Results  

- **Training**  
  The model was trained over 10 epochs using cross-entropy loss and the Adam optimizer (learning rate: 1e-4). The loss values for each epoch are shown below:

  ```
  Epoch 1/10, Loss: 3.0176  
  Epoch 2/10, Loss: 1.0653  
  Epoch 3/10, Loss: 0.8873  
  Epoch 4/10, Loss: 0.7173  
  Epoch 5/10, Loss: 0.5731  
  Epoch 6/10, Loss: 0.4719  
  Epoch 7/10, Loss: 0.4020  
  Epoch 8/10, Loss: 0.3492  
  Epoch 9/10, Loss: 0.3079  
  Epoch 10/10, Loss: 0.2730  
  ```

  The training loss consistently decreased over the epochs, indicating that the model effectively learned to map GIF features to textual descriptions.

---

### Conclusion  
The project successfully demonstrated the ability to extract meaningful visual features from GIFs using a pre-trained ResNet50 model and model the relationship between these features and the corresponding text using an RNN. The results showed a significant decrease in training loss, suggesting effective learning. Future improvements could include evaluation on test data, the use of attention mechanisms, and further hyperparameter tuning.

---
### Future Modifications
In future work, we plan to address the limitation of not being able to download the entire set of GIFs from the TGIF dataset due to computational constraints. By downloading and processing all the GIFs that could not be included in this project, the model would be exposed to a more diverse dataset. This would allow it to learn more comprehensively, leading to improved performance, reduced error rates, and more robust and clear outputs. Expanding the dataset would enhance the model's generalization ability, enabling it to better map visual features to textual descriptions.
