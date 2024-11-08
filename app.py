import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

# Function to display metrics
def display_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_pred)
    
    st.markdown(f"**📊 Model Evaluation**")
    st.write(f"**Accuracy:** {accuracy:.4f}")
    st.write(f"**Precision:** {precision:.4f}")
    st.write(f"**Recall:** {recall:.4f}")
    st.write(f"**F1 Score:** {f1:.4f}")
    st.write(f"**AUC-ROC:** {auc:.4f}")

def apply_black_red_theme():
    plt.style.use('dark_background')  # Use dark background for the plots
    plt.rcParams['axes.facecolor'] = 'black'  # Set the background color of the plot area
    plt.rcParams['axes.edgecolor'] = 'red'  # Set the edge color of the axes
    plt.rcParams['axes.labelcolor'] = 'red'  # Set the label color to red
    plt.rcParams['xtick.color'] = 'red'  # Set x-tick color to red
    plt.rcParams['ytick.color'] = 'red'  # Set y-tick color to red
    plt.rcParams['grid.color'] = 'gray'  # Set grid color to a neutral gray for contrast
    plt.rcParams['grid.linestyle'] = '--'  # Set grid lines to dashed
    plt.rcParams['figure.facecolor'] = 'black'  # Set the figure background to black
    plt.rcParams['figure.edgecolor'] = 'red'  # Set figure edge color to red


# Streamlit UI
st.title("🚀 Classification Model Trainer")

# Initialize step in session_state if not already
if 'step' not in st.session_state:
    st.session_state.step = 1  # Set initial step

def next_step():
    st.session_state.step += 1

st.markdown(
    """
    ## Welcome to the Classification Model Trainer! 🎉

    This web app allows you to train various classification models on your dataset, select features, and evaluate model performance.
    Whether you're a beginner or an experienced analyst, this tool simplifies the process of creating accurate classification models.

    **Features:**
    - Upload and preprocess your dataset
    - Select features and target variables for classification
    - Choose from multiple classification models
    - Evaluate model performance with metrics like accuracy, precision, recall, and AUC-ROC
    - Save your trained model for future use

    ⚠️ **Important:** Please upload a clean dataset with only numerical data. Convert any categorical data to numerical format (e.g., one-hot encoding) before uploading to ensure the model runs smoothly.
    """
)

# Step 1: Upload Dataset
if st.session_state.step >= 1:
    st.header("📤 Step 1: Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file is not None:
        # Load dataset
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Store the dataset in session state
        st.write("Data Preview", df.head())

    if uploaded_file is not None:
        st.button("Next Step ➡️", key="next_step_1", on_click=lambda: st.session_state.update({"step": 2}))


# Step 2: Select Features and Target Variable
if st.session_state.step >= 2:
    st.header("⚙️ Step 2: Select Features and Target Variable")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        features = st.multiselect("Select Features 📊", df.columns.tolist())
        target = st.selectbox("Select Target Variable 🎯", df.columns.tolist())

        if features and target:
            st.session_state.X = df[features]
            st.session_state.y = df[target]
            st.write("Selected Features:", features)
            st.write("Selected Target:", target)

    if features and target:
        st.button("Next Step ➡️", key="next_step_2", on_click=lambda: st.session_state.update({"step": 3}))


# Step 3: Suggested Model Based on Features and Target
if st.session_state.step >= 3:
    st.header("🧠 Step 3: Suggested Model Based on Your Data")

    if len(st.session_state.X.columns) <= 3: 
        suggested_model = "Logistic Regression"
    elif len(st.session_state.X.columns) > 3 and len(st.session_state.X.columns) <= 6:
        suggested_model = "Decision Tree Classifier"
    else:
        suggested_model = "Random Forest Classifier"

    st.write(f"Suggested Model: **{suggested_model}** based on your data.")

    model_options = ["Logistic Regression", "K-Nearest Neighbors", "SVM", "Kernel SVM", 
                     "Naive Bayes", "Decision Tree Classifier", "Random Forest Classifier"]
    model_choice = st.selectbox("Choose a Classification Model", model_options)

    st.session_state.model_choice = model_choice
    st.button("Next Step ➡️", key="next_step_3", on_click=lambda: st.session_state.update({"step": 4}))


# Step 4: Split Dataset into Train and Test Sets
if st.session_state.step >= 4:
    st.header("📈 Step 4: Split Dataset")
    split_data = st.radio("Do you want to split the data into train and test sets?", ["Yes", "No"])

    if split_data == "Yes":
        test_size = st.slider("Select test size", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(st.session_state.X, st.session_state.y, test_size=test_size, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.write(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    else:
        X_train, y_train = st.session_state.X, st.session_state.y
        st.session_state.X_train = X_train
        st.session_state.y_train = y_train
        st.write(f"Using entire dataset for training: {len(X_train)} rows")

    st.button("Next Step ➡️", key="next_step_4", on_click=lambda: st.session_state.update({"step": 5}))


# Step 5: Feature Scaling Option
if st.session_state.step >= 5:
    st.header("⚖️ Step 5: Feature Scaling Option")
    feature_scaling = st.radio("Apply Feature Scaling?", ["Yes", "No"])

    if feature_scaling == "Yes":
        st.session_state.scaler = StandardScaler()
        st.session_state.X_train = st.session_state.scaler.fit_transform(st.session_state.X_train)
        if 'X_test' in st.session_state:
            st.session_state.X_test = st.session_state.scaler.transform(st.session_state.X_test)

    st.button("Next Step ➡️", key="next_step_5", on_click=lambda: st.session_state.update({"step": 6}))



# Step 6: Find Best Parameters and Adjust Parameters (for Classification)
if st.session_state.step >= 6:
    st.header("🚀 Step 6: Find Best Parameters and Adjust Parameters")

    # Option for automatic or manual tuning
    tuning_option = st.radio("Choose tuning method:", ["Automatic Tuning", "Manual Tuning"])

    if tuning_option == "Automatic Tuning":
        # Hyperparameter tuning logic for classification models
        if st.button("Find Best Parameters"):
            with st.spinner("⏳ Searching for the best parameters... This may take a while"):
                time.sleep(1)  # Short delay to show the spinner

                param_dist = {}
                rand_search = None

                if st.session_state.model_choice == "Logistic Regression":
                    param_dist = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'saga']}
                    rand_search = RandomizedSearchCV(LogisticRegression(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                elif st.session_state.model_choice == "K-Nearest Neighbors":
                    param_dist = {'n_neighbors': range(3, 15), 'metric': ['minkowski', 'euclidean']}
                    rand_search = RandomizedSearchCV(KNeighborsClassifier(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                elif st.session_state.model_choice == "SVM":
                    param_dist = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                    rand_search = RandomizedSearchCV(SVC(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                elif st.session_state.model_choice == "Kernel SVM":
                    param_dist = {'C': [0.1, 1, 10], 'kernel': ['poly', 'rbf']}
                    rand_search = RandomizedSearchCV(SVC(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                elif st.session_state.model_choice == "Naive Bayes":
                    rand_search = GaussianNB()

                elif st.session_state.model_choice == "Decision Tree Classifier":
                    param_dist = {'max_depth': range(1, 15), 'min_samples_split': [2, 5, 10]}
                    rand_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                elif st.session_state.model_choice == "Random Forest Classifier":
                    param_dist = {'n_estimators': [10, 50, 100], 'max_depth': range(1, 15), 'min_samples_split': [2, 5, 10]}
                    rand_search = RandomizedSearchCV(RandomForestClassifier(), param_dist, n_iter=5, cv=KFold(n_splits=3), scoring='accuracy', n_jobs=-1, random_state=42)

                rand_search.fit(st.session_state.X_train, st.session_state.y_train)
                best_model = rand_search.best_estimator_

                st.session_state.model = best_model
                st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
                st.write(f"Best Model: {best_model}")
                st.write(f"Best Parameters: {rand_search.best_params_}")

    elif tuning_option == "Manual Tuning":
        # Manual parameter input section
        if st.session_state.model_choice == "Logistic Regression":
            C = st.number_input("C (Regularization strength)", 0.01, 100.0, step=0.1, value=1.0)
            solver = st.selectbox("Solver", ['liblinear', 'saga'])
            st.session_state.model = LogisticRegression(C=C, solver=solver)

        elif st.session_state.model_choice == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of neighbors (k)", 1, 20, value=5)
            metric = st.selectbox("Distance metric", ['minkowski', 'euclidean'])
            st.session_state.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

        elif st.session_state.model_choice == "SVM":
            C = st.number_input("C (Regularization parameter)", 0.01, 100.0, step=0.1, value=1.0)
            kernel = st.selectbox("Kernel", ['linear', 'rbf'])
            st.session_state.model = SVC(C=C, kernel=kernel)

        elif st.session_state.model_choice == "Decision Tree Classifier":
            max_depth = st.slider("Max depth", 1, 20, value=5)
            min_samples_split = st.slider("Min samples split", 2, 20, value=2)
            st.session_state.model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)

        elif st.session_state.model_choice == "Random Forest Classifier":
            n_estimators = st.slider("Number of estimators", 10, 200, value=100)
            max_depth = st.slider("Max depth", 1, 20, value=5)
            min_samples_split = st.slider("Min samples split", 2, 20, value=2)
            st.session_state.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)

        if st.button("Confirm Parameters"):
            st.write("Parameters set manually for the selected model.")
            st.session_state.model.fit(st.session_state.X_train, st.session_state.y_train)
    
    st.button("Next Step ➡️", key="next_step_6", on_click=lambda: st.session_state.update({"step": 7}))


# Step 7: Evaluate Model
if st.session_state.step >= 7:
    st.header("📊 Step 7: Evaluate Model")

    if st.session_state.model:
        y_pred = st.session_state.model.predict(st.session_state.X_test)

        apply_black_red_theme()

        # Display metrics
        display_metrics(st.session_state.y_test, y_pred)

        # Bar plot for Correct vs Incorrect Predictions
        st.subheader("### Correct vs Incorrect Predictions")
        correct = (st.session_state.y_test == y_pred).sum()
        incorrect = len(st.session_state.y_test) - correct

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(['Correct', 'Incorrect'], [correct, incorrect], color=['green', 'red'])
        ax.set_ylabel('Count')
        ax.set_title('Correct vs Incorrect Predictions')
        st.pyplot(fig)

        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(st.session_state.y_test, y_pred)
        
        # Plot confusion matrix using seaborn heatmap
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], 
                    yticklabels=['Actual Negative', 'Actual Positive'], cbar=False, annot_kws={"size": 16})
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        # Classification Report
        from sklearn.metrics import classification_report
        report = classification_report(st.session_state.y_test, y_pred)
        st.write("Classification Report")
        st.text(report)

        # Download the trained model
        pickle_filename = 'trained_model.pkl'
        pickle.dump(st.session_state.model, open(pickle_filename, 'wb'))

        st.download_button(
            label="Download Model",
            data=open(pickle_filename, "rb").read(),
            file_name=pickle_filename,
            mime="application/octet-stream"
        )

