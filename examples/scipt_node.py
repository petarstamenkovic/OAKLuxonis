# This code demonstrates a simple script usage

# Import libraries
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Create script node
script = pipeline.create(dai.node.Script)

# Create queues
inputQueue = script.inputs["in"].createInputQueue()
outputQueue = script.outputs["out"].createOutputQueue()

# Input the actual script
script.setScript(
    """
    while True:
        message = node.inputs["in"].get()
        # Or alternatively:
        # message = node.io["in"].get()
        node.warn("I received a message!")
        node.outputs["out"].send(message)
        # Or alternatively:
        # node.io["out"].send(message)
"""
)

# Start the pipeline
pipeline.start()

# Main loop for processing
with pipeline:
    while pipeline.isRunning():
        message = dai.ImgFrame()
        print("Sending a message")
        inputQueue.send(message)
        output = outputQueue.get()
        print("Received a message")
        time.sleep(1)
