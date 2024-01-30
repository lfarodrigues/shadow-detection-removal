# Shadow detection and removing from aerial images
import matplotlib.pyplot as plt
import cv2 
import numpy as np
from skimage.filters import threshold_multiotsu
# Constants
window_width, window_height = 800, 600  

# Function to convert an image in RGB to CIELCh color space 
def rgb_to_cielch(rgb_image):
    # Convert RGB to CIE Lab
    lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2Lab)

    # Extract Lab channels
    L, a, b = cv2.split(lab_image)

    # Convert CIE Lab to CIE LCh
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a) * 180 / np.pi  # Convert radians to degrees

    # Normalize channels if needed
    L_normalized = L / 255.0
    C_normalized = C / np.max(C)
    h_normalized = (h + 180) / 360.0

    return L_normalized,h_normalized, C_normalized

def calculate_spectral_ratio(L_channel, h_channel):
    # Calculate spectral ratio (Sr)
    Sr = (h_channel + 1) / (L_channel + 1)

    # Calculate natural logarithm of Sr (Srlog)
    Srlog = np.log(Sr + 1)

    # Define the filter matrix Bf (5x5)
    filter_matrix = np.ones((5, 5), dtype=np.float32) / 25

    # Convolve Srlog with Bf
    Srlog_filtered = cv2.filter2D(Srlog, -1, filter_matrix)

    return Srlog_filtered

def filter_lh_channels(LCh_image, kernel_size = 5):
     # Extract L and h channels
    L_channel = LCh_image[:, :, 0]
    h_channel = LCh_image[:, :, 2]

    # Apply mean filter to L channel
    L_filtered = cv2.blur(L_channel, (kernel_size, kernel_size))

    # Apply mean filter to h channel
    h_filtered = cv2.blur(h_channel, (kernel_size, kernel_size))

    # Normalize to the range [0, 1]
    cv2.normalize(L_filtered, L_filtered, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h_filtered, h_filtered, 0, 1, cv2.NORM_MINMAX)

    return L_filtered, h_filtered

def multilevel_otsu_thresholding(img, num_thresholds = 3):
    # Perform multilevel Otsu thresholding using threshold_multiotsu
    thresholds = threshold_multiotsu(img, classes=num_thresholds)
    # Get the major of the thresholds
    return thresholds


def create_binary_mask(image, threshold):
    # Apply binary thresholding to create a mask
    binary_mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)[1]
    return binary_mask

def apply_morphological_closing(binary_mask, kernel_size=3):
    # Define a kernel for morphological closing
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply morphological closing to the binary mask
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    return closed_mask

def label_connected_regions(binary_mask):
    # Ensure the binary mask is of type np.uint8
    binary_mask = binary_mask.astype(np.uint8)
    # Find connected components in the binary mask
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    return labels, stats

def calculate_illumination_ratio(unshaded_region, shaded_region, p=1):
    # Calculate overall illumination of unshaded and shaded regions
    Lunshaded = np.mean(unshaded_region) ** (1/p)
    Lshaded = np.mean(shaded_region) ** (1/p)

    # Calculate illumination ratio
    ratio = (Lunshaded - Lshaded) / Lshaded

    return ratio

def remove_shadows(image, labels, stats):
    shadow_free_image = np.zeros_like(image, dtype=np.float32)    # Extract the intensity component from the image
    intensity_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for label in range(1, len(stats)):
        # Extract the region corresponding to the current label
        region_mask = np.uint8(labels == label) * 255

        region = cv2.bitwise_and(intensity_image, intensity_image, mask=region_mask)

        cv2.imshow('shadow reg', cv2.resize(region, (window_width, window_height)))

        # Get the statistics for the current region
        left, top, width, height, area = stats[label]

        # Get a border mask by subtracting the submask from the dilated submask
        submask = region_mask
        dilated_submask = cv2.dilate(submask, np.ones((5, 5), np.uint8), iterations=1)
        border_mask = dilated_submask - submask
 
        # Ensure unshaded_borders has the same size and type as the region
        unshaded_borders = cv2.bitwise_and(intensity_image, intensity_image, mask=border_mask)
  
        cv2.imshow('unsh borders', cv2.resize(unshaded_borders, (window_width, window_height)))

        # Calculate the illumination ratio between border and shadow region
        illumination_ratio = calculate_illumination_ratio(unshaded_borders, region, p=1)

        k = 0
        #relight_shadow = ((illumination_ratio + 1)/(k*illumination_ratio+1)) * region
        k = 1

        # Assign the calculated pixels to the shadow-free image
        
        #for i in range(image.shape[0]):
         #   for j in range(image.shape[1]):
          #      if region_mask[i, j] != 0:  # Verificar se o pixel está na região da máscara
           #         shadow_free_image[i, j] = (illumination_ratio+1) * image[i, j]
        
        cv2.waitKey()
        # Multiply each pixel in the original image within the shaded region by the illumination ratio
    return shadow_free_image


def remove_sombra(imagem, labels, stats):
    #imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    shadow_free_image = np.copy(imagem)    # Extract the intensity component from the image
    intensity_img = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    for label in range(1, len(stats)):
        # Extract the region corresponding to the current label
        submask = np.uint8(labels == label) * 255

        shaded_region_org = cv2.bitwise_and(imagem, imagem, mask=submask)
        shaded_region = cv2.bitwise_and(intensity_img, intensity_img, mask=submask)

        cv2.imshow('shadow reg', cv2.resize(shaded_region, (window_width, window_height)))


        # Get a border mask by subtracting the submask from the dilated submask
        dilated_submask = cv2.dilate(submask, np.ones((5, 5), np.uint8), iterations=1)
        border_mask = cv2.subtract(dilated_submask, submask)
 
        # Ensure unshaded_borders has the same size and type as the region
        unshaded_borders = cv2.bitwise_and(intensity_img, intensity_img, mask=border_mask)
  
        cv2.imshow('unsh borders', cv2.resize(unshaded_borders, (window_width, window_height)))

        #ratio = (L_unshaded - L_shaded) / L_shaded
        #print(ratio)
        # (c) Reluzimento dos pixels
        #relighted_region = cv2.multiply(shaded_region, ratio+1)

        #cv2.imshow('relight', cv2.resize(relighted_region, (window_width, window_height)))

        L_unshaded = np.mean(unshaded_borders)
        L_shaded = np.mean(shaded_region)
        print(L_unshaded)
        print(L_shaded)
        ratio = (np.clip(abs(L_unshaded - L_shaded) / L_shaded, 0, 1)).astype(float) + 1

        print(ratio)

        for c in range(imagem.shape[2]):
            relighted_region = cv2.multiply(shaded_region_org[:, :, c], ratio)
            cv2.imshow('relighted region', cv2.resize(relighted_region, (window_width, window_height)))
            subtracted_img = cv2.subtract(shadow_free_image[:, :, c], shaded_region_org[:, :, c])
            cv2.imshow('subtracted', cv2.resize(subtracted_img, (window_width, window_height)))
            shadow_free_image[:, :, c] = cv2.add(subtracted_img, relighted_region)

        cv2.imshow('sdw free', cv2.resize(shadow_free_image, (window_width, window_height)))

        cv2.waitKey()
        # Multiply each pixel in the original image within the shaded region by the illumination ratio
    
    return shadow_free_image

image_path = 'imgs/aerial03.jpg'  # Replace with the path to your image
rgb_image = cv2.imread(image_path)

# Example usage
# Assume you have L and h channels from the previous conversion
L_normalized, h_normalized, C_normalized = rgb_to_cielch(rgb_image)[:3]

# Calculate spectral ratio
Srlog = calculate_spectral_ratio(L_normalized, h_normalized)

# Multilevel otsu
thresholds = multilevel_otsu_thresholding(Srlog * 255, num_thresholds=3)

# plot histogram
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))

#Plotting the histogram and the two thresholds obtained from multi-Otsu.
ax[1].hist(rgb_image.ravel(), bins=255)
ax[1].set_title('Histogram')
for thresh in thresholds:
    ax[1].axvline(thresh, color='r')


# Create binary max with the maximum threshold
major_threshold = np.max(thresholds)

binary_mask = create_binary_mask(Srlog * 255, major_threshold)

closed_mask = apply_morphological_closing(binary_mask)

# Label connected regions in the binary mask
labels, stats = label_connected_regions(closed_mask)

# Remove shadows from the image

enhanced = remove_sombra(rgb_image, labels, stats)

#enhanced = remove_shadows(rgb_image, labels, stats)

# Display 
cv2.imshow('Logarithm of Spectral Ratio (Srlog)', cv2.resize((Srlog * 255).astype(np.uint8), (window_width, window_height)))
cv2.imshow('Bin Mask', cv2.resize(binary_mask, (window_width, window_height)))
cv2.imshow('Closed Mask', cv2.resize(closed_mask, (window_width, window_height)))
cv2.imshow('Labels', (labels * 255 / np.max(labels)).astype(np.uint8))
cv2.imshow('Shadow-Free Image', cv2.resize(enhanced, (window_width, window_height)))

plt.subplots_adjust()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

