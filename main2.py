import torch.cuda
from ultralytics import YOLO
import cv2
import time

focal_length = 20



class ObjectDetection:
    def __init__(self, video_path, class_heights=None):
        self.video_path = video_path
        self.class_heights = class_heights if class_heights is not None else {}

        # Check if CUDA is available and set the device accordingly
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using device: ", self.device)

        # Load the YOLO model
        self.model = self.load_model()

    def load_model(self):
        model = YOLO('best.pt')
        model.fuse()
        return model

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []
        distances = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            xyxys.append(boxes.xyxy)
            confidences.append(boxes.conf)
            class_ids.append(boxes.cls)

        print("Detected classes:", [self.model.names[int(cls)] for cls in class_ids[0]])
        print("Confidences:", confidences)

        for (result, class_id) in zip(results, class_ids[0]):
            cord = result.boxes.xywh.tolist()[0]
            y, h = cord[1], cord[3]
            image_object_height = h
            class_name = self.model.names[int(class_id)].lower()  # Convert class name to lowercase
            known_height = self.class_heights.get(class_name, 0.0)
            if known_height == 0.0:
                print(f"Unknown height for class '{class_name}'. Cannot calculate distance.")
                continue
            distance = (focal_length * known_height) / image_object_height * 100
            distances.append(distance)

        print("Distances:", distances)

        for (xyxy, class_id, confidence, distance) in zip(xyxys[0], class_ids[0], confidences[0], distances):
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = self.model.names[int(class_id)]
            label = f"{class_name} {confidence:.2f} {distance:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame, xyxys, confidences, class_ids, distances

    def __call__(self):
        cap = cv2.VideoCapture(self.video_path)
        assert cap.isOpened(), "Error: Cannot open video file. Please check if the file exists."

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 640))
            original_frame = frame.copy()

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.model(frame)

            if len(results) == 0:
                cv2.imshow('Object Detection', cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            plot_img, _, _, _, _ = self.plot_bboxes(results, frame)

            cv2.imshow('Object Detection', cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            end_time = time.time()
            elapsed_time = end_time - start_time
            fps_text = f"FPS: {1 / elapsed_time:.2f}"
            print(fps_text)  # Print FPS for debugging

        cap.release()
        cv2.destroyAllWindows()



known_heights = {
    'person': 1.7,
    'tank': 2.4,
    'vehicle': 1.5,
    'truck': 3.5,
    'car': 1.5
}

# Path to the input video file
video_path = "C:/Users/user/PycharmProjects/yolo/Video6.mp4"

# Create an instance of ObjectDetection with the known heights dictionary and video path
if __name__ == "__main__":
    object_detection = ObjectDetection(video_path, class_heights=known_heights)
    object_detection()
