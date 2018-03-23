# Machine Learning Engineer Nanodegree
## Capstone Project
Sen Li  
March 22nd, 2018

## I. Definition
### Project Overview
Until a few years ago, the quality of voice telecommunications has been limited by design choices made over 100 years ago, which resulted in an 8 kHz sampling rate being used and in a practical frequency range of 300 – 3400 Hz. This so called narrowband (NB) frequency range severely limited speech quality. Recently, the industry has started to move to “HD voice” and “Ultra HD voice”, for example, the use of wideband (WB) or super-wideband (SWB) speech coders respectively, which use a sampling rate of 16 kHz or 32 kHz and correspond to a frequency range of 50-7000 Hz or 50 –14000 Hz respectively [1][2].

However, these deployments are not ubiquitous. A whole new infrastructure is needed to support these WB and SWB coders, at a substantial cost. It will likely take years before complete coverage is achieved. Until then, a significant proportion of calls will still use legacy narrowband. Further, it is likely that landline upgrades to WB or SWB will take even longer, meaning that even when the mobile networks have fully migrated to higher bandwidths, calls from landlines will still be narrowband.

Blind bandwidth extension (BBE) technology aims at solving this problem by transforming NB speech into WB or SWB speech. Typically using some form of either spectral folding or statistical modelling, the 4-8 kHz part of a speech signal is predicted from the 0-4 kHz part, to generate a signal having the general characteristics of wideband speech [3][4]. While perfect prediction cannot be expected, reasonably high quality speech can be obtained. In this project, we focused on predicting the 4-8 kHz portion of speech, usually referred to as the high-band (HB), from the 0-4 kHz portion, known as the low-band (LB).



### Problem Statement

Various approaches to BBE have been proposed and studied. Vector Quantization (VQ) codebook mapping is one of the classical method, which creates discreet mapping of speech parameters from LB to HB [5][6]. Gaussian Mixture Models(GMM) based method are used to preserve a more accurate transformation between LB and HB by modeling the speech envelope parameter continuously [7]. Hidden Markov Model (HMM) was the extension of GMM to improve the quality during speech transition by exploiting speech temporal information [8]. 

Recent advancement in neural networks learning, especially deep learning, suggested that such framework may have the potential to model more complex non-linear relationship between speech LB and HB, which leads to our proposal of this project. In this project, we studied the neural networks approach for blind bandwidth extension - in particular, from speech LB spectral features to accurately predict speech HB spectral features, such as line spectral frequencies (LSFs). Mean squared error (MSE), which is widely used for many regression modeling problems, is used as the error metric between predicted HB features and target HB features in our project.

![diagram](/image/diagram.png)

Since the BBE should be a generic speech enhancement algorithm, it should perform equally well for both male and female, and across various talkers and different languages. The ideal dataset for this project would be a multi-lingual speech database that contains multiple talkers and covers many languages. We decided to use the NTT 1994 multi-lingual corpus, containing 21 languages, 4 female and 4 male talks for each of the language [9]. Unfortunately, this speech corpus is not publicly available for free and the size of dataset is very large, we therefore extracted the speech features from the raw speech signal for the project purpose. For our test inputs, we evaluated the BBE performance on ITU P.501 British English test signal [10]. 



### Metrics
In this project, we built and implemented a speech HB spectral prediction module inside a typical BBE system based on deep neural networks. The training data is speech spectral features calculated from NB speech and WB speech respectively. The neural network model was trained based on NB spectral features as input from NTT 1994 corpus, and predicted the corresponding HB spectral features. We evaluated the model on unseen speech data from P.501 British English test speech. We adopted the MSE in feature domain as the metric to quantify model performance. We compare the predicted high-band spectral features with reference high-band spectral features extracted from the true wideband speech. The lower the score, the better the prediction model perform. the mathematic expression is, $error = (y_p  -  y) ^2$ , where $y_p$ is the predicted output, and $y$ is the reference output, or the ground truth. We established that deep neural networks based model would be able to capture more complex non-linear relationship between speech LB and HB and thus yield better prediction accuracy.



## II. Analysis

### Data Exploration
The data that we used for training and testing the models are the speech spectral shape features extracted from the raw speech waveform and its frequency domain representation through Fourier Transform. Line spectral frequency is a classical spectral representation of speech, not only because it is a very compact representation, compared to a 512 or 1024 points FFT, but also because of its noise robust numerical property that made it insensitive to coding noise. LSFs also contain valuable information regarding the stableness of the vocal tract filter. The LSF values lie between (0, pi/2) in the radius frequency domain and the values are in monotonic increasing order, indicating a stable vocal tract filter response. The Figure below shows a typical 10th order LB LSF.

![LB_LSF_values](/image/LB_LSF_values.png)

Given the interesting property of LSF values, we also investigate their distribution across training dataset. The Figure below shows the histogram of all LB LSFs features used for  training. We can observe that all the 10 dimensions of the feature are close to Gaussian distribution, except for some minor tails on both side of the dimension. As we discussed before, all the LSFs are within 0 and pi/2. Based on the data distribution, which are close to the optimal input space that most neural network operates upon, and also considered to keep the ordering property of the LSFs, we decided not to do special handling for data normalization and transformation. Even though we noticed that the mean of each dimension is not centered at 0, hopefully the biases parameters in the neural network model should be more than sufficient to compensate for this small value shift. 

![LB_LSF_hist](A:\udacity-ml-projects\capstone\image\LB_LSF_hist.png)

The HB LSFs distribution of the training dataset is close to that of LB LSFs, as shown in the Figure below. It's important to note that all the data used in these Figure after the necessary pre-processing step, described in data pre-processing section, from which all the abnormalities and outliers were removed. Since we pruned the background silence from the original speech corpus before extracting the speech features, the distribution here is truly representing the active speech.

![HB_LSF_hist](A:\udacity-ml-projects\capstone\image\HB_LSF_hist.png)



### Exploratory Visualization
Normally the LSFs comes in pairs and each pair of LSFs indicates the location of speech formant, or energy peak in frequency domain. The closer a pair of LSFs are, the more energy the formant contains, translating to a sharper peak in frequency domain. The Figure below illustrates several examples of 10th-order LB LSFs from the training dataset.

![LB_LSF_spectral_shape](/image/LB_LSF_spectral_shape.png)

Different from the LB LSFs, which contain the most of human speech energy and have more strong formant peaks, speech HB formant are flatter and contain much less energy, resulting in a much spread out distribution of LSF values - no pair of LSF stay close, as shown in the Figure below, which correspond to the three LB LSFs in the Figure above.

![HB_LSF_spectral_shape](/image/HB_LSF_spectral_shape.png)



### Algorithms and Techniques
In this project, we built and implemented four neural network based speech HB spectral prediction models for BBE system. We started by building a basic multi layer perceptions (MLP) model with only one hidden layer to evaluate the advantage from non-linear modeling compared to traditional linear regression modeling and conventional clustering approaches. We further improved the basic MLP model with more neurons for each layer in a deeper architecture. As discussed above, the input and output features are all LSFs, from our past experience with speech related problems, including the delta features in the input will help in general, to produce more accurate and smooth predictions given that fact that it bring certain context or temporal information in an implicit way. This has also been validated by us for this particular BBE prediction problem and therefore, we adopted the delta features as out standard configuration. Since we learned that temporal or context information will help with the speech HB spectral prediction, we further investigated the use of LSTM model, which is known for good at utilizing temporal information with its recursive architecture. Similar to the methodology for MLP models, we started by building a basic LSTM model with one LSTM layers and explored various potential improvement on top of it.



### Benchmark
We trained two classical benchmark models given the same input LB spectral features and the output HB spectral features for our comparison: 

- 1-best VQ codebook mapping model. 
- Linear regression model.

For the VQ codebook mapping model, we trained a codebook for LB LSFs concatenated with HB LSFs using K-Means algorithm. In the prediction phase, we took the spectral features from NB speech signal and calculated the nearest neighbor codebook entry based on the spectral feature distance to synthesize the corresponding HB spectral features. The model performance results were measured by the MSE in the spectral feature domain. 

We adopted the methodology by training the model on 1%, 10% and 100% of the training data and evaluated the model through training error, validation error and testing error respectively. The classical 1-best VQ code book mapping model provided a decent low bar to start with, given the fact that it is a more than 30 years old technology. In general, the more data for training, the better performance in validation and testing data, as you can see from the Figure below. One interesting observation is that the training error got a bit worse with 100% training data, the fact is we only evaluate the training error on the first 10k data point out of 400k training data. As more data included in the training, the clustering algorithm, such as K-Means, would normally improve for all the training data, but given a portion of training data, it might get worse. 

From the results below, with the full training data, the best MSE on validation set is 0.0097, and the best MSE on testing set is 0.0096. These results are far from perfect, but from the previous literature, it still can give descent output. Several improvement can be made by using weighted N-best clustering entries or using soft clustering algorithm such as GMM. These will not be covered by this project.



| VQ Codebook | training error | validation error | testing error |
| ----------- | -------------- | ---------------- | ------------- |
| 1%          | 0.0082         | 0.0110           | 0.0104        |
| 10%         | 0.0081         | 0.0101           | 0.0101        |
| 100%        | 0.0091         | 0.0097           | 0.0096        |



![result_VQ_codebook](/image/result_VQ_codebook.png)



For the linear regression model, we trained a model to learn the direct linear mapping function from input LB LSFs to output HB LSFs by minimizing the MSE. The optimization is done by numerical approach through gradient descent algorithm. As shown in the Figure below, the model was trained using 1%, 10%, and 100% of the data until fully converge. Training error, validation error and testing error were all getting better as more data used in the training. We can already see the power of more data by comparing this model to the conventional VQ codebook mapping model. With small set of training data (1%), the performance of linear regression model are comparable to VQ codebook model and the MSE on testing set is 0.0096. However, with 10% and 100% of the training data, the MSE on testing set improved to 0.0086 and 0.0082 respectively. Which is another good benchmark model to compare against.



| Linear Regression | training error | validation error | testing error |
| ----------------- | -------------- | ---------------- | ------------- |
| 1%                | 0.0075         | 0.0091           | 0.0096        |
| 10%               | 0.0068         | 0.0082           | 0.0086        |
| 100%              | 0.0069         | 0.0080           | 0.0082        |



![train_linear_regression](/image/train_linear_regression.png)

results

![result_linear_regression](/image/result_linear_regression.png)





## III. Methodology
### Data Preprocessing
Given that we are using the multi-lingual NTT 1994 corpus and it is not publicly available, we will need to perform pre-processing of the dataset and extract the speech features that we can directly use as the input and output training data for our project. 

The wideband speech data from the corpus is sampled at 16 kHz sampling rate and digitized into 16-bit resolution.The ITU P.341 Tx filter is applied to the wideband speech to simulate the typical Tx response in the telecommunication system before speech parameter extraction for both low-band and high-band. Given the parameterized speech data, we prepares the training and validation data with classical 10-fold cross validation scheme for training. 

The same pre-process and feature extraction procedure will also be applied to the test input speech, which is from ITU P.501 British English test signal. Since the original P.501 English test signal is sampled at 48 kHz, a 3:1 down sampling is required to convert it to wideband speech, sampled at 16 kHz and a further 2:1 down sampling is required to convert it to narrowband speech, sampled at 8 kHz. All the sampling rate conversion can be achieved using standardized ITU G.191 STL speech tools [11].

One problem still remains though, is for general speech related problems, especially for clean speech recordings, the background silence is not the point of interest. Voice activity detection (VAD) algorithms, which is implemented in many standardized speech coders, can effectively remove most of the silence within the recording, so that the feature that we extracted is truly representing the active speech. 



### Implementation and Refinement

All the neural network based models were built with Keras [12] toolkit using the Tensorflow [13] backend. Adam optimizer with 1e-3 learning rate and back propagation were used for training and optimization. We implemented two types of neural network models - MLP and LSTM

##### Multi Layer Perceptron (MLP)

Our basic MLP model used one hidden layer of 128 neurons, dropout is turned off for this model due to the fact that the total number of parameters are small (total model parameters: 3462) and from the simulation, it is not over fitting the data. We called this model basic MLP.

We further improved the basic MLP model with 256 neurons for each layer in a deeper architecture with 3 hidden layers. Due to the deeper architecture of the model and wider hidden layer, the number of total parameters reached 138502, to train it to generalize well, we introduced dropout for each of the hidden layer with a 50% drop probability. We called this model Improved MLP.

##### Long Short Term Memory (LSTM)

Since we learned that temporal information will help with the speech HB spectral prediction, we started with a basic LSTM model with 2 time steps memory, compared to standard MLP with only 1 time step, we used 128 neurons for the recursive layer, just to line up with the configuration in the basic MLP. Even though the number of parameters in LSTM model is a lot more than the corresponding MLP model. We called this model the basic LSTM.

Similar to the methodology for MLP models, we tried to improve from the basic LSTM model. We stacked more recurrent layers, but didn't see any improvement with a lot more parameter to train, we also increased the time step to 3 and deliberately delayed the input feature by 1 time step to allow the network to learn from future information, this produced marginal improvement over the basic LSTM model. We called this model improved LSTM.



## IV. Results
### Model Evaluation and Validation
##### Basic MLP model

The results of the basic MLP model can be found below, we trained the model using 1%, 10% and 100% of the training data and evaluated on training error, validation error and testing error respectively. We took special caution to make sure there are sufficient training epoch for the model with small training set to converge. The error for validation and testing data were getting lower along with more training data. The training error is slightly worse in the 100% training data case, the reason has been discussed in the benchmark VQ codebook model section. The final MSE on validation set is 0.0072 and MSE on testing set is 0.0071. This basic starter model have already demonstrated the advantage from non-linear modeling compared to traditional linear regression modeling, achieving relative improvement of 10% and 12% on validation and testing set respectively.



| Basic MLP | training error | validation error | testing error |
| --------- | -------------- | ---------------- | ------------- |
| 1%        | 0.0072         | 0.0087           | 0.0089        |
| 10%       | 0.0062         | 0.0077           | 0.0079        |
| 100%      | 0.0067         | 0.0072           | 0.0072        |



![train_mlp](/image/train_mlp.png)



![result_mlp](/image/result_mlp.png)



##### Improved MLP model

We improved the basic MLP model with 128 neurons for each layer in a deeper architecture with 3 hidden layers. Due to the increasing number of parameters to training the model, we introduced 50% dropout rate for each of the hidden layers. With the help of a deeper model, we could achieve another 4.1% and 6.9% improvement MSE over validation and testing set respectively. We also experimented with more deeper and wider MLP models and didn't observe better performance. The results are shown in the Table and Figure below.



| Improved MLP | training error | validation error | testing error |
| ------------ | -------------- | ---------------- | ------------- |
| 1%           | 0.0070         | 0.0086           | 0.0085        |
| 10%          | 0.0058         | 0.0076           | 0.0074        |
| 100%         | 0.0060         | 0.0069           | 0.0067        |



![train_mlp_improved](/image/train_mlp_improved.png)



![result_mlp_improved](/image/result_mlp_improved.png)



##### Basic LSTM model

In order to utilize temporal information from the dataset, we implemented the basic LSTM model with 1 recurrent layer containing 128 LSTM unit. The total number of parameters from the model is 77062, even less than the improved MLP model. With the help of recurrent architecture and temporal knowledge, we gained another error reduction of 5.7% on validation data and remained similar on testing data. The results are shown below.



| Basic LSTM | training error | validation error | testing error |
| ---------- | -------------- | ---------------- | ------------- |
| 1%         | 0.0073         | 0.0093           | 0.0094        |
| 10%        | 0.0055         | 0.0077           | 0.0077        |
| 100%       | 0.0057         | 0.0065           | 0.0065        |



![train_lstm](/image/train_lstm.png)



![result_lstm](/image/result_lstm.png)



##### Improved LSTM model

We also improved the LSTM model by allowing more temporal information in the input and incorporate future knowledge into the training. We observed that training error getting even lower, meaning the model itself is more powerful in modeling data, however, we only achieved marginal improvement on validation set and testing error got slightly worse, as shown below.



| Improved LSTM | training error | validation error | testing error |
| ------------- | -------------- | ---------------- | ------------- |
| 1%            | 0.0072         | 0.0092           | 0.0093        |
| 10%           | 0.0053         | 0.0078           | 0.0076        |
| 100%          | 0.0054         | 0.0064           | 0.0066        |



![train_lstm_improved](/image/train_lstm_improved.png)



![result_lstm_improved](/image/result_lstm_improved.png)



With the proper handle during the training, all the models above were aligned with our expectation except that we hoped to further boost the results with improved LSTM, but it didn't. Nevertheless, The performance has been tested and verified on unseen data, showing the robustness of all the models.



### Justification
After evaluating on all the previous models, the following Figure illustrate our final ranking on our models. Compared to the conventional benchmark models, including VQ codebook mapping and linear regression approaches, all neural network based architecture showed the strengths in outperforming the benchmark models by more accurately modelled the non-linear relationship between speech LB spectral shape and speech HB spectral shape. Given all the performance below, I would choose the Basic LSTM model as my final model, given the fact that it achieve the lowest error in test set and with less parameters compared to the improved version of itself. Given the fact that the conventional VQ codebook mapping have already produced reasonable result, our final model achieved 32% improvement on top of that, which is a convincing fact that the model should be a good candidate for this bandwidth extension problem.

![final_ranking](/image/final_ranking.png)



## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization


![BBE_frame_work](/image/BBE_frame_work.png)



![WB_Spectrogram](A:\udacity-ml-projects\capstone\image\WB_Spectrogram.png)



![NB_Spectrogram](A:\udacity-ml-projects\capstone\image\NB_Spectrogram.png)



![BBE_Spectrogram](A:\udacity-ml-projects\capstone\image\BBE_Spectrogram.png)





In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:

- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
