# Importing dependencies 
import cv2 
import torch 
import matplotlib.pyplot as plt 

# Downloading MiDaS 
midas = torch.hub.load("intel-isl/MiDaS", 'MiDaS_small')
midas.to('cpu')
midas.eval() 

# Input transformation pipeline 
transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
transform = transforms.small_transform

# Hook into OpenCV
cap = cv2.VideoCapture(0)
if not cap.isOpen(): 
     print("Cannot open camera")
     exit()
while cap.isOpened(): 
    ret, frame = cap.read() 

    # Transform input for midas 
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to('cpu')
    # Make a prediction 
    with torch.no_grad(): 
         prediction = midas(imgbatch)
         # Resize the image to fit the original image 
         prediction = torch.nn.functional.interpolate(
              prediction.unsqueeze(1),
              size = img.shape[:2], 
              mode = 'bicubic', 
              align_corners=False  
         ).squeeze()

        # Moves the tensor to CPU and converts it to a np array for matplotlib
         output= prediction.cpu().numpy()
    plt.imshow(output)
    cv2.imshow("CV2Frame", frame)
    # Pause the execution briefly to allow the plot windows to update and display the image. 
    plt.pause(0.00001)
    
    if cv2.waitKey(10) & 0xFF == ord('q'): 
         cap.release()
         cv2.destroyAllWindows() 
plt.show()
