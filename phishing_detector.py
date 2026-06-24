import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==========================================
# 1. CREATE A SIMULATED DATASET
# ==========================================
# In a real scenario, you would load a CSV using pd.read_csv('phishing_data.csv')
data = {
    'email_text': [
        "URGENT: Your bank account has been compromised. Click here to verify your identity: http://secure-update-bank.com",
        "Hey team, don't forget we have a meeting at 3 PM tomorrow in conference room B.",
        "Congratulations! You've won a $1,000 Walmart gift card. Claim your prize now at http://free-giftcards-now.info",
        "Hi Mom, just letting you know I made it home safely. Love you!",
        "Security Alert: Unauthorized login attempt on your Apple ID. Reset your password immediately: http://apple-support-auth.com/login",
        "Please review the attached quarterly financial report before Friday's board meeting.",
        "Your PayPal account is limited. Please update your billing information via this link to restore access.",
        "Are we still on for lunch today? Let me know!",
        "Final Notice: Unpaid invoice #9982. Download the attached PDF or visit http://malicious-invoice.com to pay now.",
        "Can you send me the codebase for the new feature by end of day? Thanks."
    ] * 10, # Multiplying by 10 to artificially increase dataset size for the split
    'label': ['Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe'] * 10
}

df = pd.DataFrame(data)

# ==========================================
# 2. FEATURE EXTRACTION & PREPROCESSING
# ==========================================
# Function to explicitly extract URL features by tagging them
def preprocess_email(text):
    # Convert text to lowercase
    text = text.lower()
    # Replace all URLs with a specific keyword feature "has_url_link"
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' has_url_link ', text)
    return text

# Apply preprocessing to our dataset
df['processed_text'] = df['email_text'].apply(preprocess_email)

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['label'], test_size=0.2, random_state=42)

# ==========================================
# 3. BUILD THE MACHINE LEARNING PIPELINE
# ==========================================
# A pipeline prevents data leakage and streamlines the process
model_pipeline = Pipeline([
    # TF-IDF converts text into numerical features based on word frequency and importance
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    # Logistic Regression is highly effective and fast for text classification
    ('classifier', LogisticRegression())
])

# ==========================================
# 4. TRAIN THE MODEL
# ==========================================
print("Training the model...")
model_pipeline.fit(X_train, y_train)

# ==========================================
# 5. EVALUATE THE MODEL
# ==========================================
print("\nEvaluating model on test data...")
predictions = model_pipeline.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")

# Display Confusion Matrix
print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, predictions, labels=['Safe', 'Phishing'])
matrix_df = pd.DataFrame(conf_matrix, 
                         columns=['Predicted Safe', 'Predicted Phishing'], 
                         index=['Actual Safe', 'Actual Phishing'])
print(matrix_df)

# Display Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================================
# 6. TEST WITH NEW, UNSEEN EMAILS
# ==========================================
print("\n--- Testing Custom Emails ---")
custom_emails = [
    "Dear customer, your Netflix subscription has expired. Click http://netflix-renew-sub.com to update your credit card.",
    "Hey John, attached is the presentation for tomorrow's pitch. Let me know what you think.",
    "URGENT ACTION REQUIRED: Verify your Microsoft 365 credentials immediately to avoid account deletion."
]

# Preprocess the custom emails before predicting
processed_custom = [preprocess_email(email) for email in custom_emails]
custom_predictions = model_pipeline.predict(processed_custom)

for email, prediction in zip(custom_emails, custom_predictions):
    print(f"\nEmail: '{email}'")
    print(f"Prediction: >> {prediction} <<")