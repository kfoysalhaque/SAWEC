
from sklearn.metrics import accuracy_score
import time


def evaluate_performance(model, test_data, test_labels):
    true_labels = []
    predicted_labels = []
    inference_time_list = []
    skip_first_inference = True

    for i in range(len(test_data)):
        x = test_data[i:i + 1]  # Select a single image
        start_time = time.time()  # Start timer
        predictions = model.predict(x)
        end_time = time.time()  # End timer

        # Calculate and print inference time for each image
        inference_time = end_time - start_time
        print(f"Inference time for image {i}: {inference_time} seconds")

        if skip_first_inference:
            skip_first_inference = False
            continue  # Skip the first inference time

        inference_time_list.append(inference_time)

        # Extract predicted and true labels for accuracy calculation
        predicted_label = predictions.tolist()[0]
        true_label = test_labels[i].tolist()

        predicted_labels.append(predicted_label.index(max(predicted_label)))
        true_labels.append(true_label.index(max(true_label)))

    # Calculate the average inference time
    average_inference_time = sum(inference_time_list) / len(inference_time_list)
    print(f"Average inference time for all test images: {average_inference_time} seconds")

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"The test accuracy is: {accuracy}%")