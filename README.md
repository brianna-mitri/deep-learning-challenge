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
