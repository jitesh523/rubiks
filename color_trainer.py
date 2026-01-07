import os
import pickle

import cv2

samples = {"white": [], "red": [], "green": [], "blue": [], "yellow": [], "orange": []}


def add_sample(label, pixel):
    samples[label].append(pixel)
    print(f"✔️ Added sample for {label}: {pixel} (Total: {len(samples[label])})")


def lab_pixel(frame, x, y):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    return lab[y, x]


def display_status():
    status = []
    for color, data in samples.items():
        status.append(f"{color}: {len(data)} samples")
    return " | ".join(status)


print("🧠 ML-Based Color Trainer for Rubik's Cube")
print("=" * 50)
print("Press keys 1-6 to label center pixel:")
print("1: White  | 2: Red    | 3: Green")
print("4: Blue   | 5: Yellow | 6: Orange")
print("s: Save   | q: Quit   | c: Clear all")
print("=" * 50)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

label_map = {
    ord("1"): "white",
    ord("2"): "red",
    ord("3"): "green",
    ord("4"): "blue",
    ord("5"): "yellow",
    ord("6"): "orange",
}

# Load existing samples if available
if os.path.exists("lab_classifier.pkl"):
    try:
        with open("lab_classifier.pkl", "rb") as f:
            samples = pickle.load(f)
        print("📂 Loaded existing training data")
    except:
        print("⚠️ Could not load existing data, starting fresh")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Could not read frame")
        break

    h, w, _ = frame.shape
    x, y = w // 2, h // 2

    # Draw crosshair
    cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
    cv2.line(frame, (x - 20, y), (x + 20, y), (0, 255, 0), 2)
    cv2.line(frame, (x, y - 20), (x, y + 20), (0, 255, 0), 2)

    # Display instructions
    cv2.putText(
        frame,
        "Point center at sticker and press 1-6",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        "1:White 2:Red 3:Green 4:Blue 5:Yellow 6:Orange",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Display sample counts
    status_text = display_status()
    cv2.putText(frame, status_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Color Trainer - ML-Based", frame)

    key = cv2.waitKey(1) & 0xFF

    if key in label_map:
        pix = lab_pixel(frame, x, y)
        add_sample(label_map[key], pix)

    elif key == ord("s"):
        # Save training data
        with open("lab_classifier.pkl", "wb") as f:
            pickle.dump(samples, f)
        print("✅ Saved training data to lab_classifier.pkl")

        # Show summary
        print("\n📊 Training Data Summary:")
        total_samples = 0
        for color, data in samples.items():
            count = len(data)
            total_samples += count
            status = "✅ Good" if count >= 5 else "⚠️ Need more" if count >= 3 else "❌ Too few"
            print(f"  {color.capitalize():8}: {count:2d} samples {status}")
        print(f"  Total: {total_samples} samples")

    elif key == ord("c"):
        samples = {"white": [], "red": [], "green": [], "blue": [], "yellow": [], "orange": []}
        print("🗑️ Cleared all samples")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\n👋 Color trainer finished!")
if any(len(data) > 0 for data in samples.values()):
    print("💡 Tip: Press 's' to save before quitting next time!")
