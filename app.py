from scipy.stats import f_oneway
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Load the trained Random Forest model
with open("random_forest_model.pkl", "rb") as model_file:
    best_rf_model = pickle.load(model_file)

feature_names = ['Melt temperature', 'Mold temperature', 'time_to_fill',
       'ZDx - Plasticizing time', 'ZUx - Cycle time', 'SKx - Closing force',
       'SKs - Clamping force peak value',
       'Mm - Torque mean value current cycle',
       'APSs - Specific back pressure peak value',
       'APVs - Specific injection pressure peak value',
       'CPn - Screw position at the end of hold pressure', 'SVo - Shot volume']
# Define class labels
quality_labels = {1.0: "Waste", 2.0: "Acceptable", 3.0: "Target", 4.0: "Inefficient"}

# Streamlit UI
st.title("Quality Prediction Dashboard")
st.sidebar.header("Enter Process Parameters")

# User Input for Features
user_input = []
for feature in feature_names:
    value = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Convert user input into a NumPy array
user_input_array = np.array(user_input).reshape(1, -1)

# Predict Quality Class
if st.sidebar.button("Predict Quality"):
    try:
        prediction = best_rf_model.predict(user_input_array)[0]
        prediction_proba = best_rf_model.predict_proba(user_input_array)[0]

        predicted_class = quality_labels.get(prediction, "Unknown")
        confidence = max(prediction_proba) * 100

        # Display Prediction Results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Quality Class:** {predicted_class}")
        st.write(f"**Confidence Level:** {confidence:.2f}%")

        # Feature Importance Visualization
        if hasattr(best_rf_model, "feature_importances_"):
            st.subheader("Feature Importance")
            feature_importance = best_rf_model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df = importance_df.sort_values(by='Importance', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_xlabel("Feature Importance")
            ax.set_ylabel("Features")
            ax.set_title("Feature Importance Ranking")
            st.pyplot(fig)
        else:
            st.warning("Feature importance is not available for this model.")

        # Confusion Matrix & Classification Report
        st.subheader("Model Performance Metrics")

        # Load test data if available
        try:
            test_data = pd.read_csv("test_data.csv")  # Ensure you have test data available
            y_test = test_data["quality"]  # Replace with actual column name
            X_test = test_data[feature_names]
            y_pred = best_rf_model.predict(X_test)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=quality_labels.values(), yticklabels=quality_labels.values())
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)

            scrap_rate = (y_test.value_counts(normalize=True)[1.0] * 100) if 1.0 in y_test.values else 0
            st.subheader("Production Quality Overview")
            st.write(f"**Scrap Rate:** {scrap_rate:.2f}% of total production")


            # Classification Report
            report = classification_report(y_test, y_pred, target_names=quality_labels.values(), output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

        except Exception as e:
            st.warning(f"Test data not found or invalid. Cannot compute confusion matrix. Error: {e}")
        
        
            # Get feature columns (excluding target variable)
            feature_columns = X_test.columns

            anova_results = {}
            for feature in feature_columns:
                groups = [X_test[y_test == cls][feature] for cls in y_test.unique()]
                
                if all(len(group) > 1 for group in groups):  # Ensure groups have enough data
                    stat, p_value = f_oneway(*groups)
                    anova_results[feature] = p_value
                else:
                    anova_results[feature] = None  # Skip if there's insufficient data

            # Convert results to a DataFrame
            anova_df = pd.DataFrame(list(anova_results.items()), columns=['Feature', 'P-Value'])

            # Plot ANOVA results
            plt.figure(figsize=(10, 5))
            sns.barplot(data=anova_df, x="P-Value", y="Feature", palette="viridis")
            plt.axvline(x=0.05, color="red", linestyle="--", label="Significance Threshold (0.05)")
            plt.xlabel("ANOVA P-Value")
            plt.ylabel("Feature")
            plt.title("ANOVA Feature Importance")
            plt.legend()
            plt.show()

        except Exception as e:
            st.warning(f"ANOVA visualization not available due to missing test data. Error: {e}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.sidebar.info("Enter process parameters and click 'Predict Quality' to get results.")
