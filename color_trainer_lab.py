import cv2
import numpy as np
import pickle
import os

# Initialize samples dictionary
samples = {c: [] for c in ["white", "red", "green", "blue", "yellow", "orange"]}
label_keys = {
    ord("1"): "white",
    ord("2"): "red",
    ord("3"): "green",
    ord("4"): "blue",
    ord("5"): "yellow",
    ord("6"): "orange",
}


def load_existing_samples():
    """Load existing samples if available"""
    if os.path.exists("lab_samples.pkl"):
        try:
            return pickle.load(open("lab_samples.pkl", "rb"))
        except:
            print("⚠️ Could not load existing samples, starting fresh")
    return {c: [] for c in ["white", "red", "green", "blue", "yellow", "orange"]}


def display_sample_counts(samples):
    """Display current sample counts with status"""
    status_line = []
    for color, data in samples.items():
        count = len(data)
        if count >= 10:
            status = "✅"
        elif count >= 5:
            status = "⚠️"
        else:
            status = "❌"
        status_line.append(f"{color}: {count}{status}")
    return " | ".join(status_line)


def main():
    global samples
    samples = load_existing_samples()

    print("🧠 LAB Color Trainer for Rubik's Cube (Optimized)")
    print("=" * 60)
    print("🎯 Goal: Collect 10-15 samples per color for best accuracy")
    print("📋 Controls:")
    print("  1: White  | 2: Red    | 3: Green")
    print("  4: Blue   | 5: Yellow | 6: Orange")
    print("  s: Save   | c: Clear  | q: Quit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return

    print("📂 Loaded existing samples:", display_sample_counts(samples))
    print("\n💡 Tips:")
    print("  • Vary lighting conditions between samples")
    print("  • Try different angles and distances")
    print("  • Collect samples from actual cube stickers")
    print("  • Aim for 10-15 samples per color")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Could not read frame")
            break

        H, W = frame.shape[:2]
        x, y = W // 2, H // 2

        # Draw crosshair and targeting circle
        cv2.circle(frame, (x, y), 20, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.line(frame, (x - 30, y), (x + 30, y), (0, 255, 0), 2)
        cv2.line(frame, (x, y - 30), (x, y + 30), (0, 255, 0), 2)

        # Display instructions
        cv2.putText(
            frame,
            "Point at cube sticker and press 1-6",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            "1:White 2:Red 3:Green 4:Blue 5:Yellow 6:Orange",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Display sample counts at bottom
        status_text = display_sample_counts(samples)
        cv2.putText(
            frame, status_text, (10, H - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )

        cv2.putText(
            frame,
            "s: Save | c: Clear | q: Quit",
            (10, H - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("LAB Color Trainer", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in label_keys:
            # Convert to LAB and get center pixel
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            pix = lab[y, x]
            color_name = label_keys[key]
            samples[color_name].append(pix.tolist())

            count = len(samples[color_name])
            status = (
                "✅ Excellent!"
                if count >= 10
                else "⚠️ Good, need more" if count >= 5 else "❌ Need more"
            )
            print(f"Added {color_name}: {pix} (Total: {count}) {status}")

        elif key == ord("s"):
            # Save samples
            pickle.dump(samples, open("lab_samples.pkl", "wb"))
            print("\n✅ Saved training data to lab_samples.pkl")

            # Show detailed summary
            print("\n📊 Training Data Summary:")
            total_samples = 0
            ready_colors = 0

            for color, data in samples.items():
                count = len(data)
                total_samples += count
                if count >= 10:
                    status = "✅ Excellent"
                    ready_colors += 1
                elif count >= 5:
                    status = "⚠️ OK, more recommended"
                else:
                    status = "❌ Too few"
                print(f"  {color.capitalize():8}: {count:2d} samples {status}")

            print(f"\n📈 Total: {total_samples} samples")
            print(f"🎯 Ready colors: {ready_colors}/6")

            if ready_colors >= 6:
                print("🚀 All colors have sufficient samples! Ready to train KNN.")
            elif ready_colors >= 4:
                print("👍 Good progress! Collect more samples for better accuracy.")
            else:
                print("📝 Collect more samples, especially for colors with ❌ status.")

        elif key == ord("c"):
            # Clear all samples
            samples = {c: [] for c in ["white", "red", "green", "blue", "yellow", "orange"]}
            print("🗑️ Cleared all samples")

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n👋 Color trainer finished!")
    total = sum(len(data) for data in samples.values())
    if total > 0:
        print(f"💾 You have {total} samples total. Remember to save with 's' before training!")
        print("🔄 Next step: Run 'python train_lab_knn.py' to train the classifier")


if __name__ == "__main__":
    main()
