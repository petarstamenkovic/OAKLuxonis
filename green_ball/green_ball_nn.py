# This code demonstrates how to use a custom YoloV8 model to detect a green ball.

# Import necessary libraries
import cv2
import depthai as dai
import numpy as np

# Parse YOLO data
def parse_yolo_data(tensor_data, grid_size,target_class):
    tensor = np.array(tensor_data).reshape((6, grid_size, grid_size))
    boxes = []

    for y in range(grid_size):
        for x in range(grid_size):
            # Extract the bounding box coordinates and confidence
            conf = tensor[4, y, x]
            class_id = int(tensor[5, y, x])
            if conf > 0.5 and class_id == target_class: 
                boxes.append({
                    'x' : tensor[0, y, x],
                    'y' : tensor[1, y, x],
                    'w' : tensor[2, y, x],
                    'h' : tensor[3, y, x],
                    'conf' : conf,
                    'class_id' : int(tensor[5, y, x]),
                    'grid_x' : x,
                    'grid_y' : y})
    return boxes

# Convert boxes to actual pixel coordinates
def convert_boxes_to_pixels(boxes, grid_size):
    stride = 640 / grid_size
    pixel_boxes = []

    for box in boxes:
        cx = (box['grid_x'] + box['x']) * stride
        cy = (box['grid_y'] + box['y']) * stride
        w = box['w'] * stride
        h = box['h'] * stride

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)

        pixel_boxes.append({
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2,
            'conf': box['conf'],
            'class_id': box['class_id']
        })
    return pixel_boxes

def nms(boxes, iou_threshold=0.5):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
    keep = []
    while boxes:
        box = boxes.pop(0)
        keep.append(box)
        filtered_boxes = []
        for b in boxes:
            inter_x1 = max(box['x1'], b['x1'])
            inter_y1 = max(box['y1'], b['y1'])
            inter_x2 = min(box['x2'], b['x2'])
            inter_y2 = min(box['y2'], b['y2'])

            inter_w = max(0, inter_x2 - inter_x1 + 1)
            inter_h = max(0, inter_y2 - inter_y1 + 1)
            inter_area = inter_w * inter_h

            area_box = (box['x2'] - box['x1'] + 1) * (box['y2'] - box['y1'] + 1)
            area_b = (b['x2'] - b['x1'] + 1) * (b['y2'] - b['y1'] + 1)
            union_area = area_box + area_b - inter_area

            if union_area == 0:
                iou = 0
            else:
                iou = inter_area / union_area

            if iou < iou_threshold:
                filtered_boxes.append(b)

        boxes = filtered_boxes
    return keep

# Create a pipeline
pipeline = dai.Pipeline()

# Define a camera node
cameraNode = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
cameraOut = cameraNode.requestOutput(size=(640, 640), fps=15)

# Create a Image Manipulation node to convert to RGB
manipNode = pipeline.create(dai.node.ImageManip)
manipNode.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
manipNode.setMaxOutputFrameSize(640 * 640 * 3)  # Set output size for RGB image
cameraOut.link(manipNode.inputImage)

# Define a Neural Network node - add custom YoloV8 model
nnNode = pipeline.create(dai.node.NeuralNetwork)
nnNode.setBlobPath("./result/custom_model.blob")
manipNode.out.link(nnNode.input)

# Create output queues
detectedData = nnNode.out.createOutputQueue()
originalImage = nnNode.passthrough.createOutputQueue()

# Start the pipeline
pipeline.start()

# Main loop to process frames
while pipeline.isRunning:
    rgbImage = originalImage.get()
    frame = rgbImage.getCvFrame()

    nnData = detectedData.get()

    # Check all the names of the layers in the neural network
    #print("All layers: ", nnData.getAllLayerNames())
    # Layer 1 - output1_yolov6r2
    # Layer 2 - output2_yolov6r2
    # Layer 3 - output3_yolov6r2

    # Debugging output to check the type and the attributes of NNData, documentation issues 
    #print(type(nnData))
    #print(dir(nnData)) # Lists all attributes and methods of nnData

    #out1 = nnData.getTensor("output1_yolov6r2")
    #out2 = nnData.getTensor("output2_yolov6r2")
    #out3 = nnData.getTensor("output3_yolov6r2")

    #print("Output 1: ", out1.shape)
    #print("Output 2: ", out2.shape)
    #print("Output 3: ", out3.shape) 

    boxes = []
    for output_name, size in zip(["output1_yolov6r2", "output2_yolov6r2", "output3_yolov6r2"], [80, 40, 20]):
        if nnData.hasLayer(output_name):
            tensor = nnData.getTensor(output_name).data
            boxes += parse_yolo_data(tensor, size, 0)

    pixel_boxes = []
    pixel_boxes += convert_boxes_to_pixels([b for b in boxes if b['grid_x'] < 80 and b['grid_y'] < 80], 80)
    pixel_boxes += convert_boxes_to_pixels([b for b in boxes if b['grid_x'] < 40 and b['grid_y'] < 40], 40)
    pixel_boxes += convert_boxes_to_pixels([b for b in boxes if b['grid_x'] < 20 and b['grid_y'] < 20], 20)

    # Non-
    pixel_boxes = nms(pixel_boxes)

    # Keep the best box if multiple boxes are detected
    if pixel_boxes:
        pixel_boxes = [max(pixel_boxes, key=lambda b: b['conf'])]

    # Draw bounding box and label
    for box in pixel_boxes:
        cv2.rectangle(frame, (box['x1'], box['y1']), (box['x2'], box['y2']), (0, 255, 0), 2)
        cv2.putText(frame, "Green Ball", (box['x1'], box['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Green Ball Detection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
