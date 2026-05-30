# import cv2
# from src.capture.camera import Camera
# from src.iris.recognition import IrisRecognitionSystem
#
# iris_system = IrisRecognitionSystem()
# camera = Camera()
#
# camera.open()
#
# while True:
#     result = camera.read_frame()
#
#     if not result.success:
#         continue
#
#     frame = result.frame.copy()
#
#     eyes = iris_system.extract_eye_regions(frame)
#
#     if eyes:
#         # Draw landmark points on main frame
#         for eye_name, eye_pts in eyes.items():
#             for (x, y) in eye_pts:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#
#         # Crop eyes
#         left_crop = iris_system.crop_eye(result.frame, eyes["left_eye"])
#         right_crop = iris_system.crop_eye(result.frame, eyes["right_eye"])
#
#         # Generate templates
#         if left_crop is not None:
#             left_template = iris_system.generate_iris_template(left_crop)
#             print("Left template shape:", left_template.shape)
#             cv2.imshow("Left Eye", left_crop)
#
#         if right_crop is not None:
#             right_template = iris_system.generate_iris_template(right_crop)
#             print("Right template shape:", right_template.shape)
#             cv2.imshow("Right Eye", right_crop)
#
#     cv2.imshow("Main Frame Debug", frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# camera.close()
# cv2.destroyAllWindows()

# test_iris_verify.py
from src.workflows.iris_verification import IrisVerificationWorkflow

workflow = IrisVerificationWorkflow()

result = workflow.verify("user1")

print(result)