import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc

 
@st.cache_data
def load_data():
    df = pd.read_csv('../data/WA_Fn-UseC_-Telco-Customer-Churn.csv',
                     delimiter = ",",
                     index_col=False)
    return df
    
def main():
    st.title("Streamlit Dashboard")
    data = load_data()
    page = st.sidebar.selectbox("Select a page:",["Homepage", "Exploration", "Modelling"])

    # data cleaning
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors = 'coerce')
    df2 = data.dropna(axis = 0).copy()
    if 'customerID' in df2.columns:
            df2.drop('customerID', axis=1, inplace=True)
   
    # encode variables
    for col in df2.select_dtypes(include=['object']).columns:
        le=LabelEncoder()
        le.fit(df2[col].unique())
        df2[col]=le.fit_transform(df2[col])
        
    
    if page == "Homepage":
        st.title("Homepage")
        st.text("WA_Fn-UseC_-Telco-Customer-Churn Dataset")
        st.dataframe(data)
        
    
    elif page == "Exploration":
        st.title("Exploratory Data Analysis")
        st.markdown("### Correlation Coefficient")  

        
        # Create the correlation matrix and plot
        correlation_matrix = df2.corr()
        plt.figure(figsize=(20, 16))  
        sns.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        
        # Show the plot in Streamlit
        st.pyplot(plt)  # Use st.pyplot instead of plt.show()
        plt.clf()  # Clear the figure to prevent overlap on subsequent plots

        
        # horizontal line separator
        st.markdown("---")  

        
        # Create the scatterplot
        fig1, ax = plt.subplots()
        sns.scatterplot(x='tenure', y='MonthlyCharges', hue=df2['Churn'].map({0: 'No', 1: 'Yes'}), data=df2, ax=ax)

        # plot title
        fig1.suptitle('Churn Analysis Scatter Plot among Tenure and Monthly Charges')
        
        # Display the plot in Streamlit
        st.pyplot(fig1)

        # Add explanation for the plot
        st.markdown("""
        **Purpose:** To see if there is any relationship between how long a customer has stayed (tenure) and their monthly charges.

        - **Tenure** ranges from 0 to 72 months, and **MonthlyCharges** range from approximately 20 USD to 120 USD.
        - There **doesn't appear to be a clear linear or strong correlation** between tenure and MonthlyCharges, as the data points are widely scattered across the plot.
        - Customers with varying tenures (both short and long) seem to have a **broad range of MonthlyCharges**, suggesting that the duration of a customer's subscription does not necessarily increase or decrease their monthly charges.
        - **No Clear Trend:** Customers are paying anywhere from low to high monthly charges regardless of how long they have been with the company.
        """)

        
        # horizontal line separator
        st.markdown("---")  

        
        # create scatter plot
        fig2, ax = plt.subplots()
        sns.scatterplot(x='tenure', y='TotalCharges', hue=df2['Churn'].map({0: 'No', 1: 'Yes'}) , data=df2, ax=ax)

        # plot title
        fig2.suptitle('Churn Analysis Scatter Plot among Tenure and Total Charges')
        
        # Display the plot in Streamlit
        st.pyplot(fig2)

        # Add explanation for the plot
        st.markdown("""
        **Purpose:** To explore if customers who churn tend to have higher total charges compared to those who do not.

        **Upward Trend:** Customers who churn are often found on the higher side of TotalCharges. This suggests that those who leave the service may have initially spent more, indicating potential dissatisfaction with the value received relative to their spending.

        **Spread of Charges:** There is notable variation in TotalCharges among customers at different tenure levels. This spread implies that customers with the same tenure can have significantly different charges, likely due to factors such as monthly charges, additional services, or varying discounts.

        **High TotalCharges Among Churned Customers:** The presence of churned customers with high TotalCharges indicates that these individuals may have used more services before deciding to leave, pointing to a potential disconnect between perceived value and cost.

        **Conclusion:** While there is a trend showing that customers who churn often have higher TotalCharges, the variations at each tenure level suggest that different factors contribute to this outcome. Understanding these dynamics is crucial for developing effective retention strategies.
        """)

        
        # horizontal line separator
        st.markdown("---")  

        
        # create pairplot
        fig3 = sns.pairplot(df2, vars = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'], hue='Churn')

        # plot title
        fig3.fig.suptitle('Pairplot with different classes, Colored by Churn', y=1.02)

        # Display the plot in Streamlit
        st.pyplot(fig3.fig)

        # Add explanation for the plot
        st.markdown("""
        **Purpose:**  
        - A **pairplot** allows you to visualize pairwise relationships in a dataset, helping you identify potential **correlations** and **patterns** among different variables.

        **Insights:**  
        - **Correlations:** You can identify potential **linear or non-linear correlations** between variables. For example, a **positive correlation** would appear as an upward trend in the scatter plots.  
        - **Clusters:** You may observe distinct **clusters** of data points, suggesting different groups within the dataset.  
        - **Outliers:** Unusual points that stand out from the general trend can also be identified easily.  

        **Conclusion:**  
        - **Pairplots** provide a comprehensive view of your data, making them invaluable for **exploratory data analysis**. They help to uncover relationships and insights that might warrant further investigation, guiding your analytical approach.
        """)

    
    else:
        st.title("Modelling")

        y = df2['Churn']
        x = df2.drop('Churn', axis=1)

        x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.2, random_state=42)

        scaler = StandardScaler()

        # Fit and transform the training data
        x_train_scaled = scaler.fit_transform(x_train)

        # Transform the test data
        x_test_scaled = scaler.transform(x_test)

        # Convert the scaled data back to DataFrame for easier handling
        x_train_scaled_df = pd.DataFrame(x_train_scaled, columns=x_train.columns)
        x_test_scaled_df = pd.DataFrame(x_test_scaled, columns=x_test.columns)

        # User input for PCA components
        n_components = st.slider("Select number of PCA components", min_value=1, max_value=min(x_train_scaled_df.shape[1], 20), value=16)

        # Apply PCA on the training set
        pca = PCA(n_components=None)  
        x_train_pca = pca.fit_transform(x_train_scaled_df)

        # Transform the test set using the same PCA model
        x_test_pca = pca.transform(x_test_scaled_df)

        # Check explained variance to see how much variance is captured by PCA
        explained_variance = pca.explained_variance_ratio_
        st.write("Explained variance by each component:")
        st.bar_chart(explained_variance)

        # Apply PCA with the optimal number of components
        pca_optimal = PCA(n_components=16)
        x_train_pca_optimal = pca_optimal.fit_transform(x_train_scaled_df)

        # Transform the test set using the same PCA model
        x_test_pca_optimal = pca_optimal.transform(x_test_scaled_df)


        # horizontal line separator
        st.markdown("---")  
        
        
        # Define models
        models = {
            "KNN": KNeighborsClassifier(),
            "Logistic Regression": LogisticRegression(random_state=42),
            "SVC": SVC(probability=True, random_state=42),
            "GaussianNB": GaussianNB(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "SGD": SGDClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(algorithm='SAMME', random_state=42)
        }

        # Train models and store results
        results = {}
        
        for name, model in models.items():
            try:
                model.fit(x_train_pca_optimal, y_train)
                y_pred = model.predict(x_test_pca_optimal)
                train_accuracy = model.score(x_train_pca_optimal, y_train)
                test_accuracy = accuracy_score(y_test, y_pred)
    
                # Store relevant metrics
                results[name] = {
                    'Train Accuracy': train_accuracy,
                    'Test Accuracy': test_accuracy,
                    'Predictions': y_pred
                }
            except Exception as e:
                print(f"An error occurred with {name}: {e}")


        # Display results
        results_df = pd.DataFrame(results).T  # Transpose for better layout
        results_df.reset_index(inplace=True)  # Reset index to have model names as a column
        results_df.columns = ['Model', 'Train Accuracy', 'Test Accuracy', 'Predictions']  # Rename columns

        # Display results as a table
        st.subheader("Model Training and Evaluation Results")
        st.dataframe(results_df[['Model', 'Train Accuracy', 'Test Accuracy']])

        
        # horizontal line separator
        st.markdown("---")  

        
        st.subheader('Hyperparameter Tuning Results')
                     
        # hyper parameter tuning
        param = {'C': [10], 'gamma':[0.01]} 

        SVC_grid=GridSearchCV(SVC(kernel='rbf', probability=True),
                     param,
                     refit=True,
                     verbose=0)
        
        # Retrieve the best parameters
        SVC_grid.fit(x_train_pca_optimal,y_train)

        # Retrieve the best model
        hypertuning_model = SVC_grid.best_estimator_

        # Evaluate the best model on the test set
        y_pred = hypertuning_model.predict(x_test_pca_optimal)

        # Evaluate the best model on the test set
        test_accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Model Evaluation Complete:\nThe test set accuracy of the hyperparameter-tuned SVC model is {test_accuracy:.4f}.")

        
        # horizontal line separator
        st.markdown("---")  


        # Use the hypertuned model to predict probabilities
        y_probs = hypertuning_model.predict_proba(x_test_pca_optimal)[:, 1]  # Get probabilities for the positive class

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)

        # Calculate AUC
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(fpr, tpr, color='blue', marker='o', label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Display the AUC score
        st.write(f"ROC AUC Score: {roc_auc:.4f}")

        # Add explanation for the plot
        st.markdown("""An ROC curve area (AUC) of **0.79** is generally considered to be **good**. Here’s a breakdown of what that means:

                    ### Understanding AUC:
                    1. **Definition of AUC**:
                       - The Area Under the ROC Curve (AUC) measures a model’s ability to differentiate between positive and negative classes, with a range from **0 to 1**.

                    2. **Interpretation**:
                       - **AUC = 0.5**: Indicates that the model performs no better than random chance.
                       - **AUC < 0.7**: Generally signifies poor model performance.
                       - **AUC between 0.7 and 0.8**: Reflects acceptable to good performance.
                       - **AUC between 0.8 and 0.9**: Indicates very good performance.
                       - **AUC > 0.9**: Represents excellent performance.

                    ### Specifics for AUC = 0.79 in Customer Churn Prediction:
                    - **Acceptable Discriminative Ability**: An AUC of 0.79 suggests that the model has a reasonable ability to distinguish between customers who are likely to churn and those who are likely to stay. Specifically, this means that if you randomly select one customer who churned and one who didn’t, the model will correctly rank the churn-risk customer higher 79% of the time.

                    - **Practical Considerations for Churn**: While an AUC of 0.79 is generally acceptable, it may not be sufficient for high-stakes scenarios. In the context of customer churn prediction, this score indicates that the model can be useful, but there are some considerations to keep in mind regarding false positives and false negatives:
                    - **False Positives**: These may lead to unnecessary retention efforts, potentially wasting resources on customers who are not at risk of leaving.
                    - **False Negatives**: Missing customers who are likely to churn can result in lost revenue and a negative impact on customer satisfaction.

                    Given the AUC of 0.79, the model demonstrates potential but also indicates that there may be room for improvement. Exploring additional feature engineering, hyperparameter tuning, or alternative modeling approaches could enhance its performance.

                    ### Conclusion:
                    An AUC of 0.79 is a solid indication that the model is performing reasonably well in distinguishing between customers at risk of churn and those who are not. While it reflects an acceptable predictive capability, ongoing efforts to refine the model could lead to improved performance and more effective customer retention strategies.
                    """)

        
        # horizontal line separator
        st.markdown("---")  



        # Get predictions from the best model
        best_model_name = "SVC"  
        best_model = hypertuning_model  # Use the hypertuned model

        best_model_predictions = best_model.predict(x_test_pca_optimal)
    
        # Generate the confusion matrix
        confusion_mat = confusion_matrix(y_test, best_model_predictions)

        # Display the confusion matrix
        st.write(f"Confusion Matrix for {best_model_name}:")
        st.write(confusion_mat)

        # Visualize the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'], 
            yticklabels=['Not Churned', 'Churned'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {best_model_name}')

        # Show the plot in Streamlit
        st.pyplot(plt)

        
if __name__=='__main__':
    main()