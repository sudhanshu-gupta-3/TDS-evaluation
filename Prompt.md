You are a Senior ML engineer and you are assigned to a real time traffic sign detection project, you are responsible for the whole project.

Context And Role -: Build a ML model that detects all kinds of traffic signs available in India with high accuracy and low latency rate, provide a single python working code.

Dataset - Find the best available dataset that contains all classes of traffic signs and having sufficient images per class.

Data Processing -: 
Dataset:Use the GTSRB dataset for training and testing the model. For classes having less than 1,000 images per class, use generative synthetic data techniques to bridge the gap, ensure that the model doesn't suffer from class imbalance.
Preprocessing Pipeline: Using Pandas and NumPy, implement a rigorous validation script to handle missing annotations and clean corrupt image files.
Environmental Augmentation: To meet the requirement for detecting partially visible signs, apply aggressive data augmentation. This includes random cropping, "CutMix," and simulated occlusion (shadows, tree branches, or vehicle overlays), alongside photometric transforms to mimic monsoon rain, dust storms, and night-time glare.

Input -: 
Static Images: For batch processing and forensic road audits.
Recorded Video: For post-drive analytics and fleet management reviews.
Live Stream: Real-time Webcam integration for in-vehicle assistance.

Tech Stack -: 
Inference Engine- Consider YOLOv8 as first priority, you can select any other model also if it has better performance.
Data Orchestration- Python serves as the primary language, with NumPy for tensor manipulation and Pandas for metadata analysis.
LLM Integration- OpenAI will be utilized as a natural language reasoning layer. Instead of a simple "Stop" tag, the LLM will generate contextual alerts like: "Detected 'No Entry' sign 30m ahead; please reroute to the nearest legal turn."
Interface: A high-performance dashboard built with Streamlit to render live-stream bounding boxes and real-time alert logs.



Output -: 

On detecting an sign, the system will instantly tag the sign, calculate its distance/confidence, and generate an alert message through the LLM layer, providing the driver with an actionable, safe navigation command.

Training and Evaluation- 
Training : The model will be trained over 50 epochs. To optimize computing process, we will implement an Early Stopping callback: if the validation accuracy gets constant for more than 5 consecutive epochs, the training will terminate, and the best weights will be serialized.
Evaluation Metrics: The hard target is a Testing Accuracy (mAP@.5) exceeding 90%. We will specifically evaluate the "tail" of the distribution—rare signs that appear less frequently but are critical for safety.
Latency Optimization: We aim for sub-10ms inference per frame, ensuring the system can operate at highway speeds without "frame-skipping" the detection of critical signage.
Error Handling and Documentation-: 

Handle frontend form errors gracefully.


Handle backend validation errors.


Provide structured error responses.


Log backend failures appropriately.


Requirements -: 

The model should be able to detect all kinds of traffic signs.
Accuracy in testing data must be above 90%.
The latency rate should be as low as possible.
It should be able to detect partially visible signs also.

