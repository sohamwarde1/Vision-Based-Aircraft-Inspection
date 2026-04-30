import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()

        # ArUco setup (OpenCV < 4.7)
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()

        self.subscription = self.create_subscription(
            Image,
            '/world/empty/model/sweepee_1/link/sweepee_1/wrist_3_link/sensor/sweepee_1/camera_sensor/image',
            self.image_callback,
            10
        )

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            frame = self.detect_aruco(frame)
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Conversion failed: {e}")

    def detect_aruco(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            for i, corner in enumerate(corners):
                pts = corner[0].astype(int)

                # Thick green bounding box
                cv2.polylines(
                    frame,
                    [pts],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=4
                )

                # Corner highlight dots
                for pt in pts:
                    cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)

                # "DEFECT DETECTED" label above the marker
                x_min = pts[:, 0].min()
                y_min = pts[:, 1].min()
                label = f"DEFECT DETECTED (ID: {ids[i][0]})"

                # Dark background rectangle for readability
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
                )
                bg_x1 = x_min
                bg_y1 = max(y_min - text_h - baseline - 8, 0)
                bg_x2 = x_min + text_w + 6
                bg_y2 = max(y_min - 4, 0)

                cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    label,
                    (x_min + 3, max(y_min - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

                self.get_logger().info(f"ArUco marker detected — ID: {ids[i][0]}")

        return frame


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()