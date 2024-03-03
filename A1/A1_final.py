import matplotlib.pyplot as plt
import numpy as np


def togrey(img):
    grey_img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    return grey_img

def gaussian_kernel(size, sigma):
    # size is (2k+1)x(2k+1)
    # 1 ≤ i
    # j ≤ 2k+1
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))

    kernel /= np.sum(kernel)  # Normalize the kernel

    return kernel

def conv_2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape
    output = np.zeros_like(image)
    
    # pads image
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')

    for y in range(kernel_height//2,image_height):
        for x in range(kernel_width//2,image_width):
            
            region = padded_image[(y-kernel_height//2):y+kernel_height//2 + 1, x-kernel_width//2:x+kernel_width//2 + 1]
            output[y, x] = sum(region.flatten() * kernel.flatten())
    
    return output

def gaussianblur(image, size, sigma):
    kernel = gaussian_kernel(size,sigma)
    blurred_image = conv_2d(image,kernel)
    return blurred_image

def sobel_gradient(image):
    gx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    image_height, image_width = image.shape
    new_image = np.zeros_like(image)
    edge_orientation = np.zeros_like(image)
    
    # pads image
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')

    for y in range(1,image_height):
        for x in range(1,image_width):
            region = padded_image[(y-1):(y+2), (x-1):(x+2)]
            new_image[y,x] = np.sqrt(sum(region.flatten() * gx.flatten())**2 + sum(region.flatten() * gy.flatten())**2)
            edge_orientation[y,x] = np.arctan2(sum(region.flatten() * gy.flatten()),sum(region.flatten() * gx.flatten()))
    
    return [new_image, edge_orientation]

def non_max_suppression(edge_mags, edge_ors):

    image_height, image_width = edge_mags.shape
    result = np.zeros_like(edge_mags, dtype=np.uint8)

    edge_ors_quantized = (np.round(edge_ors / (np.pi/4)) % 4).astype(int)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            current_pixel = edge_mags[y, x]

            if edge_ors_quantized[y, x] == 0:  # 0 degrees
                neighbors = (edge_mags[y, x - 1], edge_mags[y, x + 1])
            elif edge_ors_quantized[y, x] == 1:  # 45 degrees
                neighbors = (edge_mags[y - 1, x - 1], edge_mags[y + 1, x + 1])
            elif edge_ors_quantized[y, x] == 2:  # 90 degrees
                neighbors = (edge_mags[y - 1, x], edge_mags[y + 1, x])
            elif edge_ors_quantized[y, x] == 3:  # 135 degrees
                neighbors = (edge_mags[y - 1, x + 1], edge_mags[y + 1, x - 1])
            
            if current_pixel >= max(neighbors):
                result[y, x] = current_pixel

    return result

def double_threshold(image,higher,lower):
    edge_strength = np.zeros_like(image)
    image_height, image_width = image.shape
    
    for y in range(image_height):
        for x in range(image_width):
            if image[y,x] > higher:
                edge_strength[y,x] = 2
            elif image[y,x] < lower:
                pass
            else:
                edge_strength[y,x] = 2
                
    return edge_strength

def hysteresis(image,edge_strengths):
    image_height, image_width = image.shape
    result = np.zeros_like(image, dtype=np.uint8)

    for y in range(1, image_height - 1):
        for x in range(1, image_width - 1):
            if edge_strengths[y,x] == 1:
                neighbor_strengths = np.concatenate((edge_strengths[y+1,x-1:x+2],edge_strengths[y,x-1],edge_strengths[y,x+1],edge_strengths[y-1,x-1:x+2]),axis = 0).flatten()   
                if 2 in neighbor_strengths:
                    result[y,x] = image[y,x]
            elif edge_strengths[y,x] == 2:
                result[y,x] = image[y,x]

    return result

def edge_detect_pipeline(image,gblur_size,gblur_sigma,t_high,t_low):
    grey_img = togrey(image)
    blurred_img = gaussianblur(grey_img,gblur_size,gblur_sigma)
    sobel_img = sobel_gradient(blurred_img)
    nmax_sup_img = non_max_suppression(sobel_img[0],sobel_img[1])
    edge_strengths = double_threshold(nmax_sup_img,t_high,t_low)
    return hysteresis(nmax_sup_img,edge_strengths)

try:
    file_name = input("Enter image file name: ")
    img = plt.imread(file_name)

    # edge detection
    new_img = edge_detect_pipeline(img,5,2,150,50)


    fig = plt.figure(figsize=(10, 7)) 
    rows = 2
    columns = 1

    fig.add_subplot(rows, columns, 1) 
    # showing image
    plt.imshow(img) 
    plt.axis('off') 
    
    # Adds a subplot at the 2nd position 
    fig.add_subplot(rows, columns, 2) 
    plt.imshow(new_img, cmap = "gray") 
    plt.axis('off') 

    plt.show()

except FileNotFoundError:
    print("File not in directory")


