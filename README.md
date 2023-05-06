# TMSC-m7G
A transformer architecture based on multi-sense-scaled embedding features and convolutional neural network to identify RNA N7-methylguanosine sites
## Requirements
• python==3.7.0
• torch==1.8.2
• numpy==1.21.5
• pandas==1.2.4
## Usage
1. Parameter configuration
   You can modify the default parameter configuration in config.py, including the data set path, to train the model.
2. Train model
   After selecting the training mode, data set and various parameters in config.py, run main.py to output the training results and save the trained model to the generated file path (configurable).
3. Predict
   Modify your parameters and model save path in predict.py and run to see the results predicted directly by the trained model.
## Reference
This code was originally referred to (https://github.com/TearsWaiting/ACPred-LAF) and built on it with modifications and additions.
## Contact
Please feel free to contact us for any further questions.
• yujiexu321@163.com
