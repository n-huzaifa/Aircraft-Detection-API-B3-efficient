# Aircraft Classification App

This is a web application built with FastAPI that allows users to upload images of aircraft and get information about the aircraft type, generation, maximum speed, and armaments. The application uses a pre-trained deep learning model to classify the aircraft in the uploaded image.

## Features

- Upload images of aircraft
- Classify the aircraft type
- Display information about the aircraft (generation, type, max speed, armaments)
- Draw a bounding box around the detected aircraft in the image

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/n-huzaifa/Aircraft-Detection-API-B3-efficient
   ```

2. Navigate to the project directory:

   ```bash
   cd Aircraft-Detection-API-B3-efficient
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:

   ```bash
   fastapi dev .\main.py
   ```

   This will start the server at `http://localhost:8000`.

2. Open your web browser and navigate to `http://localhost:8000/docs` to access the Swagger UI.

3. Use the `/upload` endpoint to upload an image of an aircraft.

4. The server will process the image, classify the aircraft type, and return information about the aircraft, including its generation, type, maximum speed, and armaments.

5. The processed image with the bounding box and aircraft information will be displayed in the response.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - The web framework used for building the API
- [TensorFlow](https://www.tensorflow.org/) - The deep learning library used for model training and inference
- [OpenCV](https://opencv.org/) - The computer vision library used for image processing
- [Matplotlib](https://matplotlib.org/) - The plotting library used for displaying the processed image

This README provides an overview of the project, installation instructions, usage instructions, contributing guidelines, license information, and acknowledgments for the libraries used. You can customize it further based on your project's specific requirements.
