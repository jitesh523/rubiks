import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_and_prepare_data():
    """Load samples and prepare training data"""
    if not os.path.exists("lab_samples.pkl"):
        print("❌ Error: No training data found!")
        print("🔧 Please run: python color_trainer_lab.py")
        return None, None
    
    samples = pickle.load(open("lab_samples.pkl", "rb"))
    
    X, y = [], []
    for color, pixels in samples.items():
        X += pixels
        y += [color] * len(pixels)
    
    return np.array(X), np.array(y), samples

def analyze_training_data(samples):
    """Analyze and display training data statistics"""
    print("🧠 LAB KNN Trainer - Optimized")
    print("=" * 50)
    print("📊 Training Data Analysis:")
    
    total_samples = 0
    ready_colors = 0
    
    for color, pixels in samples.items():
        count = len(pixels)
        total_samples += count
        
        if count >= 10:
            status = "✅ Excellent"
            ready_colors += 1
        elif count >= 5:
            status = "⚠️ OK"
        elif count >= 3:
            status = "❌ Minimal"
        else:
            status = "💀 Too few"
        
        print(f"  {color.capitalize():8}: {count:2d} samples {status}")
    
    print(f"\n📈 Total: {total_samples} samples")
    print(f"🎯 Ready colors: {ready_colors}/6")
    
    if total_samples < 30:
        print("⚠️ Warning: Very few samples. Collect more for better accuracy!")
        return False
    elif ready_colors < 4:
        print("⚠️ Warning: Some colors have very few samples.")
        return False
    else:
        print("✅ Good training data quality!")
        return True

def train_knn_classifier(X, y, n_neighbors=3):
    """Train KNN classifier with cross-validation"""
    print(f"\n🔄 Training KNN classifier (k={n_neighbors})...")
    
    # Train the classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    knn.fit(X, y)
    
    # Cross-validation
    if len(X) >= 10:
        cv_scores = cross_val_score(knn, X, y, cv=min(5, len(X)//6), scoring='accuracy')
        print(f"📈 Cross-validation accuracy: {cv_scores.mean():.1%} (±{cv_scores.std()*2:.1%})")
    
    # Train/test split for detailed evaluation
    if len(X) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        knn_eval = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
        knn_eval.fit(X_train, y_train)
        
        y_pred = knn_eval.predict(X_test)
        print(f"📊 Test accuracy: {(y_pred == y_test).mean():.1%}")
        
        print("\n📋 Detailed Performance Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
    
    return knn

def save_model(knn):
    """Save the trained model"""
    joblib.dump(knn, "lab_knn.pkl")
    print("✅ Trained KNN model saved to lab_knn.pkl")

def test_model_predictions():
    """Test the saved model with some sample predictions"""
    try:
        knn = joblib.load("lab_knn.pkl")
        samples = pickle.load(open("lab_samples.pkl", "rb"))
        
        print("\n🧪 Testing Model Predictions:")
        print("=" * 40)
        
        # Test with one sample from each color
        for color, pixels in samples.items():
            if pixels:
                test_pixel = np.array(pixels[0]).reshape(1, -1)
                prediction = knn.predict(test_pixel)[0]
                confidence = knn.predict_proba(test_pixel).max()
                
                status = "✅" if prediction == color else "❌"
                print(f"  {color:8} → {prediction:8} ({confidence:.1%}) {status}")
        
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
        if response != 'y':
            print("👋 Please collect more samples first!")
            return
    
    # Train the classifier
    knn = train_knn_classifier(X, y)
    
    # Save the model
    save_model(knn)
    
    # Test predictions
    test_model_predictions()
    
    print("\n🚀 Training Complete!")
    print("=" * 30)
    print("📝 Next steps:")
    print("  1. Run: python lab_cube_scanner.py")
    print("  2. Or integrate detect_face_lab() into your existing scanner")
    
    print("\n💡 Usage in your code:")
    print("  import joblib")
    print("  knn = joblib.load('lab_knn.pkl')")
    print("  prediction = knn.predict([lab_pixel])[0]")

if __name__ == "__main__":
    main()
