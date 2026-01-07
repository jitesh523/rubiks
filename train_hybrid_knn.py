import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_prepare_data():
    """Load hybrid samples and prepare training data"""
    if not os.path.exists("hybrid_samples.pkl"):
        print("❌ Error: No hybrid training data found!")
        print("🔧 Please run: python color_trainer_hybrid.py")
        return None, None

    samples = pickle.load(open("hybrid_samples.pkl", "rb"))

    X, y = [], []
    for color, sample_dicts in samples.items():
        for sample_dict in sample_dicts:
            # Extract LAB values and hue from the dictionary
            lab = sample_dict["lab"]  # Should be [L, A, B]
            hue = sample_dict["hue"]  # Should be a single value

            # Create hybrid feature: [L, A, B, Hue]
            hybrid_feature = [lab[0], lab[1], lab[2], hue]
            X.append(hybrid_feature)
            y.append(color)

    return np.array(X), np.array(y), samples


def analyze_training_data(samples):
    """Analyze and display training data statistics"""
    print("🧠 Hybrid LAB+Hue KNN Trainer")
    print("=" * 50)
    print("📊 Training Data Analysis:")

    total_samples = 0
    ready_colors = 0

    for color, features in samples.items():
        count = len(features)
        total_samples += count

        if count >= 20:
            status = "✅ Excellent"
            ready_colors += 1
        elif count >= 10:
            status = "⚠️ Good"
            ready_colors += 1
        elif count >= 5:
            status = "❌ Minimal"
        else:
            status = "💀 Too few"

        print(f"  {color.capitalize():8}: {count:2d} samples {status}")

    print(f"\n📈 Total: {total_samples} samples")
    print(f"🎯 Ready colors: {ready_colors}/6")

    if total_samples < 60:
        print("⚠️ Warning: Few samples. Collect more for better accuracy!")
        return False
    elif ready_colors < 4:
        print("⚠️ Warning: Some colors have very few samples.")
        return False
    else:
        print("✅ Good training data quality!")
        return True


def train_hybrid_knn_classifier(X, y, n_neighbors=5):
    """Train hybrid KNN classifier with cross-validation"""
    print(f"\n🔄 Training Hybrid KNN classifier (k={n_neighbors})...")
    print("🎯 Features: [L, A, B, Hue] - Optimized for red/orange detection")

    # Scale the features for better KNN performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance", metric="euclidean")
    knn.fit(X_scaled, y)

    # Cross-validation
    if len(X) >= 10:
        cv_scores = cross_val_score(knn, X_scaled, y, cv=min(5, len(X) // 6), scoring="accuracy")
        print(f"📈 Cross-validation accuracy: {cv_scores.mean():.1%} (±{cv_scores.std()*2:.1%})")

    # Train/test split for detailed evaluation
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        knn_eval = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights="distance", metric="euclidean"
        )
        knn_eval.fit(X_train, y_train)

        y_pred = knn_eval.predict(X_test)
        print(f"📊 Test accuracy: {(y_pred == y_test).mean():.1%}")

        print("\n📋 Detailed Performance Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Special focus on red/orange confusion
        print("\n🔍 Red/Orange Analysis:")
        red_orange_mask = np.isin(y_test, ["red", "orange"])
        if np.any(red_orange_mask):
            ro_true = y_test[red_orange_mask]
            ro_pred = y_pred[red_orange_mask]
            ro_accuracy = (ro_true == ro_pred).mean()
            print(f"Red/Orange accuracy: {ro_accuracy:.1%}")

            # Show confusion between red and orange specifically
            red_indices = ro_true == "red"
            orange_indices = ro_true == "orange"

            if np.any(red_indices):
                red_as_orange = np.sum(ro_pred[red_indices] == "orange")
                red_total = np.sum(red_indices)
                print(f"Red misclassified as orange: {red_as_orange}/{red_total}")

            if np.any(orange_indices):
                orange_as_red = np.sum(ro_pred[orange_indices] == "red")
                orange_total = np.sum(orange_indices)
                print(f"Orange misclassified as red: {orange_as_red}/{orange_total}")

    return knn, scaler


def save_model(knn, scaler):
    """Save the trained hybrid model and scaler"""
    model_data = {"model": knn, "scaler": scaler}
    joblib.dump(model_data, "hybrid_knn.pkl")
    print("✅ Trained Hybrid KNN model and scaler saved to hybrid_knn.pkl")


def test_model_predictions():
    """Test the saved hybrid model with sample predictions"""
    try:
        model_data = joblib.load("hybrid_knn.pkl")
        knn = model_data["model"]
        scaler = model_data["scaler"]
        samples = pickle.load(open("hybrid_samples.pkl", "rb"))

        print("\n🧪 Testing Hybrid Model Predictions:")
        print("=" * 45)

        # Test with one sample from each color
        for color, sample_dicts in samples.items():
            if sample_dicts:
                # Extract LAB and hue from the sample dictionary
                sample_dict = sample_dicts[0]
                lab = sample_dict["lab"]
                hue = sample_dict["hue"]
                test_feature = np.array([[lab[0], lab[1], lab[2], hue]])

                # Scale the feature
                test_feature_scaled = scaler.transform(test_feature)

                prediction = knn.predict(test_feature_scaled)[0]
                confidence = knn.predict_proba(test_feature_scaled).max()

                status = "✅" if prediction == color else "❌"
                print(f"  {color:8} → {prediction:8} ({confidence:.1%}) {status}")

        # Test a few more samples for red/orange specifically
        print("\n🎯 Additional Red/Orange Tests:")
        for color in ["red", "orange"]:
            if color in samples and len(samples[color]) > 1:
                for i in range(min(3, len(samples[color]))):
                    sample_dict = samples[color][i]
                    lab = sample_dict["lab"]
                    hue = sample_dict["hue"]
                    test_feature = np.array([[lab[0], lab[1], lab[2], hue]])

                    # Scale the feature
                    test_feature_scaled = scaler.transform(test_feature)

                    prediction = knn.predict(test_feature_scaled)[0]
                    confidence = knn.predict_proba(test_feature_scaled).max()
                    status = "✅" if prediction == color else "❌"
                    print(f"  {color} #{i+1:2d} → {prediction:8} ({confidence:.1%}) {status}")

    except Exception as e:
        print(f"⚠️ Could not test model: {e}")


def main():
    # Load and prepare data
    result = load_and_prepare_data()
    if result is None:
        return

    X, y, samples = result

    # Analyze training data
    data_quality_ok = analyze_training_data(samples)

    if not data_quality_ok:
        print("\n🤔 Continue training anyway? (y/n): ", end="")
        response = input().lower()
        if response != "y":
            print("👋 Please collect more samples first!")
            return

    # Train the hybrid classifier
    knn, scaler = train_hybrid_knn_classifier(X, y)

    # Save the model
    save_model(knn, scaler)

    # Test predictions
    test_model_predictions()

    print("\n🚀 Hybrid Training Complete!")
    print("=" * 35)
    print("📝 Next steps:")
    print("  1. Update your cube scanner to use hybrid_knn.pkl")
    print("  2. Use features: [L, A, B, Hue] for prediction")

    print("\n💡 Usage in your code:")
    print("  import joblib")
    print("  knn = joblib.load('hybrid_knn.pkl')")
    print("  # Convert LAB to include Hue: [L, A, B, H]")
    print("  prediction = knn.predict([hybrid_feature])[0]")


if __name__ == "__main__":
    main()
