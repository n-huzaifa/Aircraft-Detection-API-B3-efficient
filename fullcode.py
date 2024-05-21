import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def inference_single_image(model, image_path, plane_details):
    img = cv2.imread(image_path)
    img_copy = img.copy()

    # Resize the image to match the input size of the model
    img = cv2.resize(img, (224, 224))

    # Preprocess the image
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)

    # Perform inference
    prediction = model.predict(img_tensor)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class]
    plane_detail = plane_details[predicted_class_name]

    # Add a bounding box (a rectangle around the entire image)
    cv2.rectangle(img_copy, (0, 0), (img_copy.shape[1], img_copy.shape[0]), (0, 255, 0), 2)

    # Display the image with the bounding box, predicted class, and plane details
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title(f"{predicted_class_name}\n{plane_detail}", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.show()

    
# Load the pre-trained model
model_path = "./finalsave.h5"
model = load_model(model_path)

class_names = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,"f-35",20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39]
plane_details = {
    class_names[0]: "Generation: 4th\nType: Attack Aircraft\nMax Speed: 0.56\nArmaments: Yes",
    class_names[1]: "Generation: 4th \nType: Transport Aircraft\nMax Speed: 0.72\nArmaments: No",
    class_names[2]: "Generation: N/A\nType: Amphibious Aircraft\nMax Speed: 0.46\nArmaments: No",
    class_names[3]: "Generation: 2nd \nType: Attack Aircraft\nMax Speed: 0.9\nArmaments: Yes",
    class_names[4]: "Generation: 4th \nType: Bomber\nMax Speed: 1.25\nArmaments: Yes",
    class_names[5]: "Generation: 5th \nType: Stealth Bomber\nMax Speed: 0.95\nArmaments: Yes",
    class_names[6]: "Generation: 2nd \nType: Bomber\nMax Speed: 0.86\nArmaments: Yes",
    class_names[7]: "Generation: N/A \nType: Amphibious Aircraft\nMax Speed: 0.64\nArmaments: No",
    class_names[8]: "Generation: 2nd \nType: Transport Aircraft\nMax Speed: 0.58\nArmaments: No",
    class_names[9]: "Generation: 4th \nType: Transport Aircraft\nMax Speed: 0.74\nArmaments: No",
    class_names[10]: "Generation: 2nd \nType: Transport Aircraft\nMax Speed: 0.75\nArmaments: No",
    class_names[11]: "Generation: 2nd \nType: Early Warning\nMax Speed: 0.6\nArmaments: No",
    class_names[12]: "Generation: 4th \nType: Multirole Fighter\nMax Speed: 2\nArmaments: Yes\n",
    class_names[13]: "Generation: 3rd \nType: Fighter\nMax Speed: 2.23\nArmaments: Yes",
    class_names[14]: "Generation: 4th \nType: Fighter\nMax Speed: 2.34\nArmaments: Yes",
    class_names[15]: "Generation: 4th \nType: Fighter\nMax Speed: 2.5\nArmaments: Yes",
    class_names[16]: "Generation: 4th \nType: Multirole Fighter\nMax Speed: 2\nArmaments: Yes",
    class_names[17]: "Generation: 4th \nType: Multirole Fighter\nMax Speed: 1.8\nArmaments: Yes",
    class_names[18]: "Generation: 5th \nType: Stealth Multirole\nMax Speed: 2.25\nArmaments: Yes",
    class_names[19]: "Generation: 5th \nType: Stealth Multirole\nMax Speed: 1.6\nArmaments: Yes",
    class_names[20]: "Generation: 4th \nType: Stealth Attack\nMax Speed: .92\nArmaments: Yes",
    class_names[21]: "Generation: 5th \nType: Stealth Fighter\nMax Speed: 2\nArmaments: Yes",
    class_names[22]: "Generation: 4th \nType:Multirole Fighter \nMax Speed: 2.2\nArmaments: Yes",
    class_names[23]: "Generation: N/A \nType: UAV\nMax Speed: 0.45\nArmaments: Yes",
    class_names[24]: "Generation: 4th \nType: Interceptor \nMax Speed: 2.35\nArmaments: Yes",
    class_names[25]: "Generation: 4th \nType: Multirole Fighter\nMax Speed: 2.2\nArmaments: Yes",
    class_names[26]: "Generation: N/A \nType: UAV\nMax Speed: 0.6 \nArmaments: No",
    class_names[27]: "Generation: 4th \nType: Multirole Fighter \nMax Speed: 1.8\nArmaments: Yes",
    class_names[28]: "Generation: 3rd \nType: Reconnaissance\nMax Speed: 3.3\nArmaments: No",
    class_names[29]: "Generation: 4th \nType: Fighter Bomber\nMax Speed: 1.8\nArmaments: Yes",
    class_names[30]: "Generation: 5th \nType: Stealth Fighter \nMax Speed: 2\nArmaments: Yes",
    class_names[31]: "Generation: 4th \nType: Multirole Fighter\nMax Speed: 1.3\nArmaments: Yes",
    class_names[32]: "Generation: 2nd \nType: Bomber\nMax Speed: 0.83\nArmaments: Yes",
    class_names[33]: "Generation: 4th \nType: Bomber\nMax Speed: 2.05\nArmaments: Yes",
    class_names[34]: "Generation: N/A \nType: Reconnaissance\nMax Speed: 0.67\nArmaments: No",
    class_names[35]: "Generation: N/A \nType: Amphibious Aircraft\nMax Speed: 0.49\nArmaments: No",
    class_names[36]: "Generation: N/A \nType: Tiltrotor\nMax Speed: 0.6\nArmaments: Yes",
    class_names[37]: "Generation: 2nd \nType: Bomber\nMax Speed: 0.96\nArmaments: Yes",
    class_names[38]: "Generation: 3rd \nType: Bomber\nMax Speed: 3.1\nArmaments: No",
    class_names[39]: "Generation: 4th \nType: Stealth Fighter\nMax Speed: 1.8\nArmaments: Yes",
    
}

# Example usage
image_path = "./aircraft.jpg"
inference_single_image(model, image_path, plane_details)