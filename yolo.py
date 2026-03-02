# coding=utf-8
import cv2
import depthai as dai
import numpy as np
import rospy
import message_filters
from geometry_msgs.msg import Pose, Pose2D
from geometry_msgs.msg import Vector3Stamped
from geometry_msgs.msg import PoseStamped
from object_detection_msgs.msg import BoundingBox
from object_detection_msgs.msg import BoundingBoxes


import sensor_msgs
from sensor_msgs.msg import Image
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32

rospy.init_node("tracker_node",anonymous=True)
rate = rospy.Rate(30)
window_pub  = rospy.Publisher('/bboxes_pub',BoundingBoxes,queue_size=10)


numClasses = 6
model = dai.OpenVINO.Blob("best1_openvino_2022.1_6shave.blob")
dim = next(iter(model.networkInputs.values())).dims
W, H = dim[:2]

output_name, output_tenser = next(iter(model.networkOutputs.items()))
if "yolov6" in output_name:
    numClasses = output_tenser.dims[2] - 5
else:
    numClasses = output_tenser.dims[2] // 3 - 5

labelMap = [
    # "class_1","class_2","..."
    "class_%s" % i
    for i in range(numClasses)
]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("image")
xoutNN.setStreamName("nn")

# Properties
camRgb.setPreviewSize(W, H)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Network specific settings
detectionNetwork.setBlob(model)
detectionNetwork.setConfidenceThreshold(0.5)

# Yolo specific parameters
detectionNetwork.setNumClasses(numClasses)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([])
detectionNetwork.setAnchorMasks({})
detectionNetwork.setIouThreshold(0.5)

# Linking
camRgb.preview.link(detectionNetwork.input)
camRgb.preview.link(xoutRgb.input)
detectionNetwork.out.link(xoutNN.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    imageQueue = device.getOutputQueue(name="image", maxSize=4, blocking=False)
    detectQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def drawText(frame, text, org, color=(255, 255, 255), thickness=1):
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness + 3, cv2.LINE_AA
        )
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness, cv2.LINE_AA
        )

    def drawRect(frame, topLeft, bottomRight, color=(255, 255, 255), thickness=1):
        cv2.rectangle(frame, topLeft, bottomRight, (0, 0, 0), thickness + 3)
        cv2.rectangle(frame, topLeft, bottomRight, color, thickness)

    def displayFrame(name, frame):
        color = (128, 128, 128)
        for detection in detections:
            bbox = frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
 
            drawText(
                frame=frame,
                text=labelMap[detection.label],
                org=(bbox[0] + 10, bbox[1] + 20),
            )
            drawText(
                frame=frame,
                text=f"{detection.confidence:.2%}",
                org=(bbox[0] + 10, bbox[1] + 35),
            )
            drawRect(
                frame=frame,
                topLeft=(bbox[0], bbox[1]),
                bottomRight=(bbox[2], bbox[3]),
                color=color,
            )
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        imageQueueData = imageQueue.tryGet()
        detectQueueData = detectQueue.tryGet()

        if imageQueueData is not None:
            frame = imageQueueData.getCvFrame()

        if detectQueueData is not None:
            detections = detectQueueData.detections
        
            track_id_out=BoundingBoxes()
            i=0
            for detection in detections:
                bbox = frameNorm(
                    frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
                )
                tracker_out= BoundingBox()
                tracker_out.xmin = int(bbox[0])
                tracker_out.ymin = int(bbox[1])
                tracker_out.xmax = int(bbox[2])
                tracker_out.ymax = int(bbox[3])
                tracker_out.probability =detection.confidence
                tracker_out.id = int(i)
                tracker_out.Class = labelMap[detection.label]
                track_id_out.header.stamp=rospy.Time.now()
                track_id_out.bounding_boxes.append(tracker_out)
                i=i+1
            window_pub.publish(track_id_out)


        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord("q"):
            break
        rate.sleep()

