# deep-learning-challenge
Utlizing data from 34,000 organizations that have previosly received funding from Alphabet Soup, build a neural network model to help Alphabet Soup select applicants with the best chances of success. This is a binary classification for money used successfully or not.

## Results
### Data Preprocessing
Target variable: "IS_SUCCESSFUL" (binary: money not used successfully is 0 while 1 is sucessfully)
Features: 
  - APPLICATION_TYPE: Alphabet Soup application type (categorical)
  - AFFILIATION: Affiliated sector of industry (categorical)
  - CLASSIFICATION: Government organization classification (categorical)
  - USE_CASE: Use case for funding (categorical)
  - ORGANIZATION: Organization type (categorical)
  - STATUS: Active status (categorical)
  - INCOME_AMT: Income classification (categorical)
  - SPECIAL_CONSIDERATIONS: Special considerations for application (categorical)
  - ASK_AMT: Funding amount requested (numerical)

The "optimized" models (in attempt to improve them) did the following:
- STATUS was removed since 99.99% of the data had the same status value
- ASK_AMT had a right skew (skewness value: 72.41) so applied log before StandardScaler for scaling
- Categorical columns used OneHotEncoder instead of pandas .get_dummies()
- Categorical preprocessing was fitted on the training data only

### Compiling, Training, and Evaluating the Model
Initial model was composed of:
- 1 hidden layer
    - Nuerons: 5
    - Activation: relu
- Output layer
    - Nuerons: 1
    - Activation: sigmoid

Results:  
![image](https://github.com/user-attachments/assets/be8e9c09-6fa8-4dc4-8853-24f53ea2a7a0)  
![image](https://github.com/user-attachments/assets/bbbf595e-7d2f-4e6b-a04d-ba1f271a58c0)

Following attempts at "optimization" involved keras.tuner and giving various options to find the "best" one. Desired accuracy of 75% was not achieved. Here are some of the results.  
Optimization model 1 (used Random Search):  
![image](https://github.com/user-attachments/assets/ed7b6f8e-599a-4d37-9196-77fe1b20db00)  
Results:  
![image](https://github.com/user-attachments/assets/f133018f-e295-41a0-9fb9-9e858eb34332)  
![image](https://github.com/user-attachments/assets/c0f39325-ebd1-425f-a7e0-c7980b7539ee)  

Optimization model 2 (used Bayesian Optimization):  
![image](https://github.com/user-attachments/assets/cdf330df-a26b-4cc8-9b15-b38095b212f2)  
Results:  
![image](https://github.com/user-attachments/assets/90612cc1-05e6-4624-975f-8d8a2b7d76fb)  
![image](https://github.com/user-attachments/assets/4e2d1cf5-5520-44b3-97f4-511fdc4adcc6)  

Optimization model 3 (used Hyperband):  
![image](https://github.com/user-attachments/assets/d0fbf81f-f495-4680-bffe-f34e87115400)  
Results:  
![image](https://github.com/user-attachments/assets/9611c9db-7a9b-4c07-8c74-b26cb64985be)  
![image](https://github.com/user-attachments/assets/053428e9-c656-47a7-ae69-0f07b64757d3)  

