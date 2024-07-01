
from ultralytics import YOLO
import cv2

focal_length = 8
known_height = 0.22


class ObjectDetection:
    def __init__(self, capture_index=0):
        self.capture_index = capture_index

        self.device = 'cpu'
        print("Using device: ", self.device)


        self.model = self.load_model()

    def load_model(self):
        model = YOLO('faces.pt')
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

            if len(result.boxes) == 0:
                print("No bounding boxes detected.")
                continue

            cord = result.boxes.xywh.tolist()[0]
            h = cord[3]
            image_object_height = h
            print("Image Object Height:", image_object_height)
            distance = (focal_length * known_height) / (image_object_height / 100)
            distances.append(distance)


        for (xyxy, class_id, confidence, distance) in zip(xyxys[0], class_ids[0], confidences[0], distances):
            x1, y1, x2, y2 = map(int, xyxy)
            class_name = self.model.names[int(class_id)]
            label = f"{class_name} {confidence:.2f} {distance:.2f}m"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame, xyxys, confidences, class_ids, distances

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Error: Cannot open camera. Please check if camera is connected."

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.model(frame)

            if len(results) == 0:
                cv2.imshow('Object Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            plot_img, xyxys, class_ids, confidences, distances = self.plot_bboxes(results, frame)

            cv2.imshow('Object Detection', cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    object_detection = ObjectDetection()
    object_detection()


