**# LSTM-based-stock-trend-analysis

Project Objective - To evaluate the performance and predict the return of AAPL and NVDA stocks using the LSTM machine learning (ML) technique)**

**Project Summary **
The study aims to evaluate the performance and predict the return of Apple Inc. and NVIDIA Corp stocks. The study employed Long Short-Term Memory (LSTM) as the sole ML method to make predictions based on 2,517 rows of AAPL (stock ticker symbol for Apple Inc.) and NVDA Corp (stock ticker symbol for Apple Inc. for NVIDIA) stocks, totaling 5,034 data points extracted from Yahoo Finance using the yfinance library in Python. The distribution of data sets was divided into a training set and a test set. The training dataset makes up 95% of the total dataset, while the testing dataset makes up 5%. The data underwent a series of pre-processing processes, where the testing data was used for prediction and the training of the data for 60 days before the next day's prediction.

The analytical process was divided into two parts. Part 1 explores the data (exploratory data analysis - EDA), which includes trend analysis, pinpointing the downtrends and uptrends of the stock charts vide moving averages (MA) technical indicator as well as noise detection based on the returns generated from the price history of both stocks. The EDA concludes with a summary of the basic statistics, namely; mean, standard deviation, median (50th percentile) as well as skewness and kurtosis to check whether the data distribution is asymmetric and has extreme value cases (outlier). The importance of EDA is to understand the basis of normalisation or scaling of the data, to allow for accurate prediction. The association between AAPL and NVDA stock return is moderately positive, given the value of 0.55, indicating a high positive association. 

The performance of the LSTM was evaluated using the RMSE and R2 values. The R2 shows how much dispersion or variation of the adjusted close price is explained by the open, high, low, and close price. In other words, the AAPL data fits the LSTM by approximately 78% when compared to 85% for NVDA. The predictions are derived from the test data (5%), and NVIDIA (figure 7b) performs better than Apple (figure 7a) for a risk-loving investor who is focused on return. This is largely indicated by the yellow lines that run concurrently with the red lines (validated cases). For NVDA, the yellow lines rise above the red lines in most points areas in the trend line showing a higher prediction than anticipated whereas the predicted values of AAPL are somewhat volatile and noisy showing too many up and down movements and an almost nonlinear path. Based on the findings, it is evident that the LSTM model is efficient. Furthermore, it offers highly precise predictions. Traders and investors might derive advantages by attentively considering the forecasts.

**Definition**

**Machine learning (ML)** is a subfield of artificial intelligence (AI) research that focuses on the development and study of statistical algorithms that can learn from data, generalise what they have learned to new data, and carry out tasks independently, all without the need for human interaction 
A **stock market** is a financial market where individuals can engage in the purchase and sale of publicly listed company stocks, which is a type of marketplace.

**Long short-term memory (LSTM)** are a type of artificial neural network (ANN) that is meant to process sequential input, such as time series, audio, or text. Long short-term memory networks are also known as LSTM networks. Particularly useful for processing data that has long-term dependencies, which means that the result at a specific time step is dependent on information from earlier time steps, it is especially beneficial for processing data that has long-term dependencies.


**Table 1. LSTM Elements**
Elements	Description	Equation	 i

| Elements  | Description | Equation |
| ------------- | ------------- |------------- |
| Input Gate  | The flow of data is controlled and channeled into the memory cell  | input gate = σ (Wi*[ht — 1, xt] + bi)	(1)  |
| Forget Gate  | The flow of data is controlled and channeled out of the memory cell  | forget gate = σ (Wf *[ht — 1, xt] + bf)	(2)  |
| Output Gate  | The output of the memory cell is controlled by other parts of the network  | outputgate = σ (Wo*[ht — 1, xt] + bo)	(3)  |
| Cell State  | The information is stored in the memory cell  | memory cell = ft*ct — 1 + it*tanh (Wc*[ht — 1, xt] + bc)	(4) |
| Hidden State  | The LSTM output unit is employed to pass information to the next unit in the LSTM as well as to make predictions  | hiddenstate = ot*tanh(ct)	(5) |


**Figure 1. Flow chart of the proposed LSTM prediction process using Python**
![image](https://github.com/user-attachments/assets/ed119fb4-0ba7-4279-8167-253c5138168c)


**Evaluation Metrics **
The root means square error (RMSE) will be used as a performance measure to check how well the trained ML models did. A variety of classification measures are available for use in determining how well an ML algorithm performs (Choudhary and Gianey, 2017). To sort these models according to how well they perform, the three most powerful metrics are usually used. The metrics used are RMSE and R2, however, for the purpose of emphasis, F-score, and ROC (Receiver Operator Characteristic) are also noted down as very important metrics for analysis (Nousi et al. 2019). Equations (2) and (3) represent the mathematical formula for Accuracy and F1_score:

Accuracy=((TRP+TRN))/((TRP+TRN+FAP+FAN))                                       [2]

F1_score=(2(Precision*Recall))/((Precision+Recall))                                       [3]

While ROC_AUC, F1_score, and accuracy are helpful metrics for evaluation, they are insufficient for certain issues. Classification issues also have two more well-known metrics: recall and precision (Ntakaris et al. 2019). Also included below are the expressions for Precision and Recall:

Recall=((TP))/((TR+FN))                                                        [4]

Precision=((TP))/((TP+FP))                                                 [5]

Additionally, the use of this tool involves the use of a confusion matrix to provide a summary of the performance of each machine learning model. Machine learning provides several indicators, including False Positives (FP), True Positives (TP), False Negatives (FN), and True Negatives (TN), which contribute to a comprehensive comprehension of the predictions made by machine learning (Lin et al. 2021). False positives refer to instances where the model's prediction is accurate but the actual sample is not. On the other hand, true positives indicate that both the model's prediction and the actual sample are accurate. False negatives, on the other hand, indicate instances where the model's prediction is inaccurate but the actual sample is accurate. Lastly, true negatives indicate instances where both the model's prediction and the actual sample are misleading.

R2 - Examining the R2 score of a model is an efficient approach for determining the degree to which the model is a good match for the data that it is analysing. To perform the computation, first, add up all of the squared differences that exist between the actual values and the projected values, then divide this total by the squared variances that exist between the mean of the actual numbers and the actual numbers, and then remove the result from 1. With the help of this metric, the correctness of the model is evaluated, and a number that is closer to 1 indicates that the model is more accurate (Soleymani and Paquet, 2020). Hence, this research study will specifically employ root mean squared error (RMSE) and R2 to assess the effectiveness of the prediction.

**Dataset**
This study utilised a total of two unique stocks. Table 3b below displays the selected technology equities. The historical stock values for each stock, spanning a period of 10 years, were obtained by downloading them from Yahoo Finance. The stocks are the biggest movers of the market as of March 6, 2024 (Yahoo Finance, 2024). The features stated in Table 3a were included in each download for each respective day starting 16 April 2014.

**Table 2. Selected input feature variables for ML models**
| Feature  | Description | 
| ------------- | ------------- |
| Open  | Opening price  |
| Close  | Closing price  | 
| High  | The highest prices in a single trading day  | 
| Low  | The lowest prices in a single trading day  | 
| Adj. Close  | The closing price after adjustments for all applicable dividends and split distribution  | 
| Volume  | The number of trades in a single trading day  | 


**Table 3. The selected technology stock data**
| Stock Ticker | Datasets | 
| ------------- | ------------- |
| AAPL  | 2,517  |
| NVDA  | 2,517  |

**Data Preprocessing **
The distribution of data sets was divided into a training set and a test set. The training dataset makes up 95% of the total dataset, while the testing dataset makes up 5%. The data underwent a series of pre-processing processes.

The total data record as communicated in Table 4b is 5,034 which comprises of the same number of rows for AAPL (2,517 rows) and NVDA (2,517 rows).
This study includes a subset of the moving average (MA) technical indicators. These indicators were chosen during the process of optimising hyper-parameters while they were being considered for inclusion in this study. To assist the algorithms in evaluating the relevance of these technical indicators and gaining knowledge of them, the values of each indicator were converted into either a '1' or a '-1' depending on whether the indicator indicated an upward or negative trend at the current time step, respectively. It is important to note that the meaning of each continuous MA process result differs depending on the particular technical indicator that under consideration. 



