import os
import pickle

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


def train_knn_classifier(n_neighbors=3):
    """Train a KNN classifier from saved LAB color samples"""

    if not os.path.exists("lab_classifier.pkl"):
        print("❌ Error: No training data found. Run color_trainer.py first!")
        return None

    # Load training data
    with open("lab_classifier.pkl", "rb") as f:
        samples = pickle.load(f)

    print("🧠 Training ML Color Classifier")
    print("=" * 40)

    # Prepare training data
    X = []
    y = []

    print("📊 Training Data Summary:")
    total_samples = 0
    for label, pixels in samples.items():
        count = len(pixels)
        total_samples += count
        status = "✅ Good" if count >= 5 else "⚠️ Few" if count >= 3 else "❌ Too few"
        print(f"  {label.capitalize():8}: {count:2d} samples {status}")

        for pix in pixels:
            X.append(pix)
            y.append(label)

    print(f"  Total: {total_samples} samples")

    if total_samples < 18:  # At least 3 samples per color
        print("⚠️ Warning: Very few training samples. Collect more for better accuracy!")

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\n🔄 Training KNN classifier (k={n_neighbors})...")

    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    knn.fit(X, y)

    # Cross-validation to estimate accuracy
    if total_samples >= 10:
        cv_scores = cross_val_score(knn, X, y, cv=min(5, total_samples // 6))
        accuracy = cv_scores.mean()
        print(f"📈 Cross-validation accuracy: {accuracy:.1%} (±{cv_scores.std()*2:.1%})")

    # Save the trained model
    joblib.dump(knn, "knn_model.pkl")
    print("✅ Trained and saved KNN model to knn_model.pkl")

    return knn


def test_classifier():
    """Test the trained classifier"""
    if not os.path.exists("knn_model.pkl"):
        print("❌ No trained model found. Train first!")
        return

    # Load model and test data
    knn = joblib.load("knn_model.pkl")

    with open("lab_classifier.pkl", "rb") as f:
        samples = pickle.load(f)

    X_test = []
    y_test = []

    for label, pixels in samples.items():
        for pix in pixels:
            X_test.append(pix)
            y_test.append(label)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Predict
    y_pred = knn.predict(X_test)

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred))


def predict_color(lab_pixel):
    """Predict the color of a single LAB pixel"""
    if not os.path.exists("knn_model.pkl"):
        print("❌ No trained model found. Train first!")
        return None

    knn = joblib.load("knn_model.pkl")
    prediction = knn.predict([lab_pixel])[0]
    confidence = knn.predict_proba([lab_pixel]).max()

    return prediction, confidence


if __name__ == "__main__":
    print("🎯 ML-Based Color Classification System")
    print("=" * 50)

    # Train the classifier
    model = train_knn_classifier()

    if model is not None:
        print("\n🧪 Testing classifier on training data...")
        test_classifier()

        print("\n💡 Usage:")
        print("  from lab_classifier import predict_color")
        print("  color, confidence = predict_color(lab_pixel)")

        print("\n🚀 Ready to use in cube scanner!")
