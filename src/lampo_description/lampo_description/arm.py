#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


# Joint positions for UP and DOWN states
UP_POSITION = {
    "sweepee_1/shoulder_lift_joint": -1.5,
    "sweepee_1/elbow_joint": 0.3,
}

DOWN_POSITION = {
    "sweepee_1/shoulder_lift_joint": -0.2,
    "sweepee_1/elbow_joint": 0.6,
}

# All joints in the controller (non-moving joints hold at 0.0)
ALL_JOINTS = [
    "sweepee_1/shoulder_pan_joint",
    "sweepee_1/shoulder_lift_joint",
    "sweepee_1/elbow_joint",
    "sweepee_1/wrist_1_joint",
    "sweepee_1/wrist_2_joint",
    "sweepee_1/wrist_3_joint",
]

HOLD_POSITION = 0.0        # Position for joints not being moved
MOVE_DURATION_SEC = 3      # Seconds per transition


class ArmUpDown(Node):
    def __init__(self):
        super().__init__("arm_up_down")

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/sweepee_1/joint_trajectory_controller/follow_joint_trajectory",
        )

        self._going_up = True  # Start by moving UP

        self.get_logger().info("Waiting for action server...")
        self._action_client.wait_for_server()
        self.get_logger().info("Action server ready. Starting up-down motion.")

        self._send_next_goal()

    def _build_goal(self, target: dict) -> FollowJointTrajectory.Goal:
        """Build a FollowJointTrajectory goal for the given target joint positions."""
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ALL_JOINTS

        point = JointTrajectoryPoint()
        point.positions = [
            target.get(joint, HOLD_POSITION) for joint in ALL_JOINTS
        ]
        point.time_from_start = Duration(sec=MOVE_DURATION_SEC, nanosec=0)

        goal.trajectory.points = [point]
        return goal

    def _send_next_goal(self):
        target = UP_POSITION if self._going_up else DOWN_POSITION
        direction = "UP" if self._going_up else "DOWN"
        self.get_logger().info(f"Moving {direction}: shoulder_lift={target['sweepee_1/shoulder_lift_joint']}, elbow={target['sweepee_1/elbow_joint']}")

        goal = self._build_goal(target)
        send_goal_future = self._action_client.send_goal_async(goal)
        send_goal_future.add_done_callback(self._goal_accepted_callback)

    def _goal_accepted_callback(self, future):
        goal_handle: ClientGoalHandle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by action server.")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)

    def _goal_result_callback(self, future):
        result = future.result()

        if result.status == 4:  # SUCCEEDED
            self.get_logger().info("Goal reached. Switching direction.")
        else:
            self.get_logger().warn(f"Goal finished with status: {result.status}. Continuing anyway.")

        # Toggle direction and send next goal
        self._going_up = not self._going_up
        self._send_next_goal()


def main(args=None):
    rclpy.init(args=args)
    node = ArmUpDown()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted. Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()