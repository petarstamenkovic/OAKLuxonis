# This code demonstrates how to use the System Logger node in a DepthAI pipeline.

import depthai as dai
import cv2

# Create a pipeline
pipeline = dai.Pipeline()

# Create a System Logger node
logger = pipeline.create(dai.node.SystemLogger)
logger.setRate(1)

# Create an output queue for the logger
loggerOut = logger.out.createOutputQueue()

# Start the pipeline
pipeline.start()

# Main loop to process frames
while pipeline.isRunning():
    # Get the log message from the output queue
    logMessage = loggerOut.get()
    
    # Print the log message
    print("Log Message:", logMessage)

