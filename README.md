# Inter-IIT-PS-24
Report on GIF Feature Extraction and Description Generation

Anshuman Nivas (EEE’26 Group 17)
Manoj Sunuguri (EEE’26 Group 13)
Introduction
The task undertaken involved solving a problem that required associating GIF data with textual descriptions. This was accomplished by leveraging a pre-trained ResNet50 model for feature extraction from the GIFs, followed by modeling the relationship between the GIF frames and the corresponding descriptions using an RNN-based architecture. The goal was to build a model capable of predicting the textual description of a GIF based on its extracted visual features.
1. Feature Extraction using ResNet50
The initial step in the approach involved extracting relevant features from the GIFs using a pre-trained ResNet50 model. The following steps were involved:
Frame Extraction
GIFs contain multiple frames, and not all frames contribute equally to the understanding of the sequence. Therefore, the model developer implemented a method to extract 16 important frames from each GIF. This was done using a sampling strategy that calculated a step size based on the total number of frames in each GIF. The frames were then preprocessed by resizing them to 224x224 pixels and normalizing them to match the input requirements of ResNet50.
Feature Extraction
A pre-trained ResNet50 model, with its fully connected layer removed, was used to extract features from each frame. The output from the penultimate layer of ResNet50, a 2048-dimensional feature vector, represented each frame.
Positional Encoding
Since GIFs are sequences of frames, capturing the order of the frames is crucial for understanding temporal relationships. To achieve this, positional encoding was applied to the extracted features using sine and cosine functions. This provided the model with additional information about the order in which the frames appeared.
Saving Features
The extracted features, combined with positional encoding, were saved as .npy files. These files served as the input for the next stage, where the relationship between the GIFs and their textual descriptions was modeled.
2. Dataset Preparation
Once the features were extracted, the dataset preparation stage began. A custom PyTorch dataset class (GIFDataset) was created to pair the GIF features with their corresponding descriptions.
Loading GIF Features
The features stored in .npy files were loaded from the directory. Each file contained the feature matrix for a particular GIF, with the positional encoding already applied.
Loading Descriptions
The textual descriptions corresponding to each GIF were stored in separate text files. The dataset loader was programmed to read these descriptions, which would be used for training the model.
Tokenization
To process the text descriptions, the BERT tokenizer (BertTokenizer) was employed. This tokenizer converted the text into tokens, padding or truncating the sequences to a fixed length of 128 tokens. The resulting tokenized descriptions were then paired with the corresponding GIF features.
3. Model Architecture
The primary model used in this approach was a Recurrent Neural Network (RNN)-based architecture designed to map GIF features to textual descriptions. The architecture involved the following components:
RNN for GIF Features
The ResNet features extracted from the GIF frames were passed through an RNN. The RNN was responsible for capturing temporal dependencies between frames, allowing the model to learn patterns in the visual sequences.
Text Embedding
The textual descriptions were embedded using a randomly initialized embedding layer. This layer transformed the tokenized descriptions into dense vector representations.
Fusion of GIF and Text Features
The model combined the output of the RNN (which processed the GIF features) with the embedded textual descriptions. This was done using a simple element-wise addition, though more complex fusion techniques could be considered in future work.
Output
The final output of the model was a sequence of predicted tokens, which aimed to reconstruct the textual description of the GIF. The model was trained to minimize the difference between the predicted tokens and the actual tokens in the textual description.
4. Training and Results
The model was trained over 10 epochs using a standard cross-entropy loss function, with the Adam optimizer employed to minimize the loss. The learning rate was set to 1e-4.
Training Loss
The training loss values over the 10 epochs were as follows:
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

The consistent decrease in the loss over the training epochs indicated that the model was learning to map GIF features to textual descriptions effectively. By the end of the 10th epoch, the loss had dropped to 0.2730, showing significant improvement from the initial value of 3.0176.
Conclusion
The approach taken in this project successfully demonstrated the ability to extract meaningful features from GIFs using ResNet50 and learn the relationship between these visual features and corresponding textual descriptions using an RNN. The results, as evidenced by the decreasing training loss, suggest that the model was able to effectively learn this mapping.


